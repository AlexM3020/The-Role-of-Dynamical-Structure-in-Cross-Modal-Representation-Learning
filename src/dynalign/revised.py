"""Configurable, memory-first experiment engine. Legacy experiment APIs are unchanged.

No result directory, model checkpoint, CSV, plot, or archive is created unless
the caller explicitly supplies output_dir. Notebook display persistence is a
separate editor setting, outside this module's control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform
import time
from typing import Callable
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F

from .experiments import environment, save_json
from .metrics import effective_rank, linear_cka, paired_bootstrap, procrustes_test_error, ridge_test_r2
from .worlds import (
    circle_coordinates, diagnostics, image_table, observe, observation_roughness,
    sensor_table, spectral_embedding, training_observations, transition_family,
)


@dataclass(frozen=True)
class FitSettings:
    steps: int = 80
    checkpoints: tuple[int, ...] = (40, 80)
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    objective: str = "vicreg"
    invariance_weight: float = 25.0
    variance_weight: float = 25.0
    covariance_weight: float = 1.0
    identity_weight: float = 25.0
    variance_target: float = 1.0
    epsilon: float = 1e-4
    clip_grad: float = 5.0
    log_every: int = 40
    checkpoint_every: int = 100

    def __post_init__(self):
        if self.objective not in {"vicreg", "cov_identity"}:
            raise ValueError("objective must be vicreg or cov_identity")
        if self.steps < 1 or self.batch_size < 2 or self.lr <= 0:
            raise ValueError("Positive steps/lr and batch_size >= 2 are required")
        if min(self.invariance_weight, self.variance_weight, self.covariance_weight,
               self.identity_weight, self.weight_decay) < 0:
            raise ValueError("Loss weights and weight decay must be nonnegative")
        if min(self.variance_target, self.epsilon, self.clip_grad, self.log_every,
               self.checkpoint_every) <= 0:
            raise ValueError("Targets, epsilon, clipping and intervals must be positive")
        if any(s < 1 or s > self.steps for s in self.checkpoints):
            raise ValueError("Every evaluation checkpoint must be within the training budget")


@dataclass(frozen=True)
class EvalSettings:
    calibration_repeats: int = 8
    test_repeats: int = 16
    ridge: float = 1e-3
    collapse_variance: float = 1e-6
    collapse_rank_fraction: float = 0.5
    batch_size: int = 512

    def __post_init__(self):
        if min(self.calibration_repeats, self.test_repeats) < 2 or self.batch_size < 1:
            raise ValueError("Use >=2 repetitions per state and a positive batch size")
        if self.ridge <= 0 or self.collapse_variance < 0 or not 0 < self.collapse_rank_fraction <= 1:
            raise ValueError("Invalid evaluation regularization or collapse settings")


@dataclass(frozen=True)
class SyntheticSpec:
    experiment: str = "alignment_collapse"
    phase: str = "development"
    world: int = 100
    repeat: int = 0
    master_seed: int = 20261108
    condition: str = "local"
    n_states: int = 32
    n_pairs: int = 512
    sensor_width: int = 12
    noise: float = 0.05
    aliased_b: bool = False
    pixels: bool = False
    image_side: int = 16
    image_noise: float = 0.03
    architecture: str = "mlp"
    width: int = 64
    dim: int = 4
    regime: str = "baseline"
    fit: FitSettings = field(default_factory=FitSettings)
    evaluation: EvalSettings = field(default_factory=EvalSettings)

    def __post_init__(self):
        if self.condition not in {"local", "scrambled", "iid"}:
            raise ValueError("Unknown transition condition")
        if self.architecture not in {"linear", "mlp"}:
            raise ValueError("Synthetic vector architecture must be linear or mlp")
        if min(self.world, self.repeat, self.master_seed) < 0 or self.n_pairs < 2:
            raise ValueError("Seeds must be nonnegative and n_pairs >= 2")
        if self.n_states < 8 or not 1 <= self.dim < self.n_states - 1:
            raise ValueError("Invalid state count or embedding dimension")
        if min(self.noise, self.image_noise) < 0 or min(self.width, self.sensor_width) < 1:
            raise ValueError("Invalid sensor or network settings")


@dataclass
class RunBundle:
    rows: list[dict] = field(default_factory=list)
    histories: list[dict] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    subject_rows: list[dict] = field(default_factory=list)

    @property
    def table(self):
        return pd.DataFrame(self.rows)

    @property
    def history(self):
        return pd.DataFrame(self.histories)

    @property
    def subjects(self):
        return pd.DataFrame(self.subject_rows)


def derived_seed(master: int, world: int, repeat: int, stream: int) -> int:
    return int(np.random.SeedSequence([master, world, repeat, stream]).generate_state(1)[0])


def select_device(requested="auto", threads=2):
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select cpu or auto")
    torch.set_num_threads(threads)
    return device


def make_encoder(shape, architecture, width, dim, seed):
    torch.manual_seed(seed)
    if architecture == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(int(np.prod(shape)), dim))
    if architecture == "mlp" and len(shape) == 1:
        return nn.Sequential(nn.Linear(shape[0], width), nn.SiLU(),
                             nn.Linear(width, width), nn.SiLU(), nn.Linear(width, dim))
    if architecture == "image_cnn" and len(shape) == 3:
        return nn.Sequential(nn.Conv2d(shape[0], 8, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
                             nn.Conv2d(8, 16, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)),
                             nn.Flatten(), nn.Linear(256, width), nn.SiLU(), nn.Linear(width, dim))
    if architecture in {"temporal_cnn", "order_invariant"} and len(shape) == 2:
        kernel = 5 if architecture == "temporal_cnn" else 1
        bins = 4 if architecture == "temporal_cnn" else 1
        return nn.Sequential(nn.Conv1d(shape[0], width, kernel, padding=kernel // 2), nn.SiLU(),
                             nn.Conv1d(width, width, kernel, padding=kernel // 2), nn.SiLU(),
                             nn.AdaptiveAvgPool1d(bins), nn.Flatten(), nn.Linear(width * bins, dim))
    raise ValueError(f"Unsupported architecture/shape: {architecture}/{shape}")


def temporal_loss(a, b, cfg: FitSettings):
    invariance = F.mse_loss(a, b)
    ac, bc = a - a.mean(0), b - b.mean(0)
    ca, cb = ac.T @ ac / (len(a) - 1), bc.T @ bc / (len(b) - 1)
    variance = (F.relu(cfg.variance_target - torch.sqrt(torch.diag(ca) + cfg.epsilon)).mean()
                + F.relu(cfg.variance_target - torch.sqrt(torch.diag(cb) + cfg.epsilon)).mean()) / 2
    off = ~torch.eye(a.shape[1], dtype=torch.bool, device=a.device)
    covariance = (ca[off].square().sum() + cb[off].square().sum()) / a.shape[1]
    target = torch.eye(a.shape[1], device=a.device) * cfg.variance_target**2
    identity = ((ca - target).square().sum() + (cb - target).square().sum()) / (2 * a.shape[1])
    if cfg.objective == "vicreg":
        loss = cfg.invariance_weight * invariance + cfg.variance_weight * variance + cfg.covariance_weight * covariance
    else:
        loss = cfg.invariance_weight * invariance + cfg.identity_weight * identity
    return loss, {"invariance": invariance, "variance": variance,
                  "covariance": covariance, "covariance_identity": identity}


def embeddings(model, observations, device, batch_size=512):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            x = torch.as_tensor(observations[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model(x).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def fit_encoder(model, pairs, cfg, seed, device, evaluate: Callable, checkpoint=None):
    """evaluate receives a frozen model; test metrics never select updates or weights."""
    first, second = pairs
    if first.shape != second.shape or len(first) < 2:
        raise ValueError("Expected >=2 same-shaped temporal observation pairs")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    x, y = torch.from_numpy(first), torch.from_numpy(second)
    history, snapshots, start, previous_seconds = [], {}, 0, 0.0
    if checkpoint is not None and Path(checkpoint).exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved["settings"] != asdict(cfg) or saved["seed"] != seed:
            raise ValueError("Checkpoint configuration mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["sampler"].cpu())
        history, snapshots, start, previous_seconds = saved["history"], saved["snapshots"], saved["step"], saved["seconds"]
    if not snapshots:
        snapshots[0] = evaluate(model)
    started = time.perf_counter()
    checkpoints = set(cfg.checkpoints) | {cfg.steps}
    for step in range(start + 1, cfg.steps + 1):
        model.train()
        indices = torch.randint(len(x), (cfg.batch_size,), generator=rng)
        loss, pieces = temporal_loss(model(x[indices].to(device)), model(y[indices].to(device)), cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()
        if step % cfg.log_every == 0 or step in checkpoints:
            history.append({"step": step, "loss": float(loss.detach().cpu()),
                            **{k: float(v.detach().cpu()) for k, v in pieces.items()}})
        if step in checkpoints:
            snapshots[step] = evaluate(model)
        if checkpoint is not None and (step % cfg.checkpoint_every == 0 or step == cfg.steps):
            checkpoint = Path(checkpoint)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            state = {"settings": asdict(cfg), "seed": seed, "step": step,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "sampler": rng.get_state(), "history": history, "snapshots": snapshots,
                     "seconds": previous_seconds + time.perf_counter() - started}
            temp = checkpoint.with_suffix(".tmp")
            torch.save(state, temp)
            temp.replace(checkpoint)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return snapshots, history, previous_seconds + time.perf_counter() - started


def rank_diagnostics(h, evaluation: EvalSettings):
    covariance = np.cov(h, rowvar=False)
    values = np.maximum(np.linalg.eigvalsh(np.atleast_2d(covariance))[::-1], 0)
    rank = effective_rank(h)
    variance = float(np.var(h, axis=0).mean())
    return {"eigenvalues": values.tolist(), "effective_rank": rank, "variance": variance,
            "collapse": bool(variance < evaluation.collapse_variance
                             or rank < h.shape[1] * evaluation.collapse_rank_fraction)}


def state_noise_diagnostics(h, n_states):
    if h.ndim != 2 or n_states < 2 or len(h) % n_states != 0 or len(h) < 2 * n_states:
        raise ValueError("Expected complete balanced state blocks with at least two repeats")
    repetitions = len(h) // n_states
    blocks = h.reshape(repetitions, n_states, -1)
    means = blocks.mean(0)
    within = float(blocks.var(0, ddof=1).sum(1).mean())
    between_observed = float(means.var(0).sum())
    # Finite repeated observations make the state-mean variance upward biased.
    between_corrected = between_observed - (1 - 1 / n_states) * within / repetitions
    return {"state_mean_variance_raw": between_observed,
            "state_mean_variance_corrected": between_corrected,
            "within_state_noise_variance": within,
            "estimated_signal_fraction": max(0, between_corrected) / max(max(0, between_corrected) + within, 1e-15)}


def posterior_target(observations, table, targets, noise, batch_size=256):
    """Bayes conditional mean under the known uniform Gaussian sensor mixture.

    Privileged diagnostic, never supplied to an encoder or selected using test labels.
    Exact aliasing is averaged at noise=0 rather than assigning an arbitrary state.
    """
    prototypes = table.reshape(len(table), -1).astype(float)
    values = observations.reshape(len(observations), -1).astype(float)
    output = []
    for start in range(0, len(values), batch_size):
        distance = ((values[start:start + batch_size, None] - prototypes[None]) ** 2).sum(-1)
        if noise == 0:
            weights = np.isclose(distance, distance.min(1, keepdims=True), atol=1e-10, rtol=1e-8).astype(float)
            weights /= weights.sum(1, keepdims=True)
        else:
            log_weights = -distance / (2 * noise**2)
            weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
        output.append(weights @ targets)
    return np.concatenate(output)


def attainable_reference(table, noise, targets, repeats, seed):
    states = np.tile(np.arange(len(table)), repeats)
    x = observe(table, states, noise, seed)
    predicted = posterior_target(x, table, targets, noise)
    truth = targets[states]
    mse = float(np.mean((predicted - truth)**2))
    baseline = float(np.mean((truth - targets.mean(0))**2))
    return {"bayes_mc_mse": mse, "bayes_mc_r2": 1 - mse / baseline if baseline > 1e-12 else np.nan}


def _synthetic_inputs(spec):
    def seed(stream):
        return derived_seed(spec.master_seed, spec.world, spec.repeat, stream)

    world_seed = derived_seed(spec.master_seed, spec.world, 0, 1)
    family, _ = transition_family(spec.n_states, world_seed)
    p = family[spec.condition]
    tables = [image_table(spec.n_states, spec.image_side) if spec.pixels else sensor_table(
        spec.n_states, derived_seed(spec.master_seed, spec.world, 0, 2), width=spec.sensor_width),
        sensor_table(spec.n_states, derived_seed(spec.master_seed, spec.world, 0, 3),
                     aliased=spec.aliased_b, width=spec.sensor_width)]
    noise = [spec.image_noise if spec.pixels else spec.noise, spec.noise]
    cal_states = np.tile(np.arange(spec.n_states), spec.evaluation.calibration_repeats)
    test_states = np.tile(np.arange(spec.n_states), spec.evaluation.test_repeats)
    calibration = [observe(t, cal_states, s, seed(300 + m)) for m, (t, s) in enumerate(zip(tables, noise))]
    test = [observe(t, test_states, s, seed(400 + m)) for m, (t, s) in enumerate(zip(tables, noise))]
    return p, tables, noise, calibration, test, cal_states, test_states


def run_synthetic(specs, *, device="auto", threads=2, output_dir=None):
    """Execute an explicit list. All settings are recorded; no automatic tuning/caching."""
    device = select_device(device, threads)
    metadata = {**environment(), "engine": "revised-memory-first", "device": device,
                "threads": threads, "python": platform.python_version(),
                "storage": "memory" if output_dir is None else str(output_dir)}
    bundle = RunBundle(metadata=metadata)
    for index, spec in enumerate(specs):
        if not isinstance(spec, SyntheticSpec):
            raise TypeError("Pass SyntheticSpec objects with explicit settings")
        plan = asdict(spec)
        identity = {"spec": plan, "environment": metadata}
        run_id = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        bundle.plans.append({"run_id": run_id, **plan})
        print(f"[{index + 1}/{len(specs)}] {spec.regime}, {spec.condition}, world={spec.world}, N={spec.n_pairs}, {spec.architecture}-{spec.width}", flush=True)
        p, tables, noise, calibration, test, cal_states, test_states = _synthetic_inputs(spec)
        spectral, gap = spectral_embedding(p, spec.dim)
        targets = {"coordinate": circle_coordinates(spec.n_states),
                   "quotient": circle_coordinates(spec.n_states, 2),
                   "next_mean": p @ circle_coordinates(spec.n_states)}
        if gap > 1e-8:
            targets["spectral"] = spectral
        all_snapshots, elapsed, parameters = [], 0.0, []
        for m, table in enumerate(tables):
            architecture = "image_cnn" if spec.pixels and m == 0 else spec.architecture
            model = make_encoder(table.shape[1:], architecture, spec.width, spec.dim,
                                 derived_seed(spec.master_seed, spec.world, spec.repeat, 100 + m)).to(device)
            parameters.append(sum(v.numel() for v in model.parameters()))
            def evaluate(current, m=m):
                return {"cal": torch.from_numpy(embeddings(current, calibration[m], device, spec.evaluation.batch_size)),
                        "test": torch.from_numpy(embeddings(current, test[m], device, spec.evaluation.batch_size))}
            pairs = training_observations(p, table, spec.n_pairs, noise[m],
                                          derived_seed(spec.master_seed, spec.world, spec.repeat, 200 + m))
            checkpoint = None if output_dir is None else Path(output_dir) / run_id / f"encoder_{m}.pt"
            snapshots, history, seconds = fit_encoder(
                model, pairs, spec.fit, derived_seed(spec.master_seed, spec.world, spec.repeat, 500 + m),
                device, evaluate, checkpoint,
            )
            all_snapshots.append(snapshots)
            elapsed += seconds
            bundle.histories.extend({"run_id": run_id, "modality": m, **item} for item in history)
        for step in sorted(all_snapshots[0]):
            ca, ta = (all_snapshots[0][step][k].cpu().numpy() for k in ("cal", "test"))
            cb, tb = (all_snapshots[1][step][k].cpu().numpy() for k in ("cal", "test"))
            row = {k: v for k, v in plan.items() if k not in {"fit", "evaluation"}}
            row.update({"run_id": run_id, "step": step, "cka": linear_cka(ta, tb),
                        "procrustes_error": procrustes_test_error(ca, cb, ta, tb),
                        "run_seconds_including_evaluation": elapsed,
                        "parameters_a": parameters[0], "parameters_b": parameters[1], **diagnostics(p, spec.dim)})
            permutation = np.random.default_rng(derived_seed(spec.master_seed, spec.world, spec.repeat, 600)).permutation(len(tb))
            row["shuffled_cka"] = linear_cka(ta, tb[permutation])
            for label, cal, h, table in (("a", ca, ta, tables[0]), ("b", cb, tb, tables[1])):
                row.update({f"{key}_{label}": value for key, value in rank_diagnostics(h, spec.evaluation).items()})
                row.update({f"{key}_{label}": value for key, value in state_noise_diagnostics(h, spec.n_states).items()})
                row[f"roughness_{label}"] = observation_roughness(p, table)
                for target_name, target in targets.items():
                    row[f"{target_name}_r2_{label}"] = ridge_test_r2(cal, target[cal_states], h, target[test_states], spec.evaluation.ridge)
            row["any_collapse"] = row["collapse_a"] or row["collapse_b"]
            bundle.rows.append(row)
        if output_dir is not None:
            persist_bundle(bundle, output_dir)
    return bundle


def paired_contrasts(table, *, metric="cka", group_by=("regime", "n_pairs", "step"),
                     unit="world", condition="condition", baseline="local", comparison="scrambled",
                     draws=5000, seed=0):
    """Pair repetitions first, then average within world/subject; report omissions."""
    if table.empty:
        return pd.DataFrame()
    group_by = list(group_by)
    keys = [*group_by, unit, "repeat", condition]
    if table.duplicated(keys).any():
        raise ValueError("Duplicate condition rows: add the varying settings to group_by")
    wide = table.set_index(keys)[metric].unstack(condition)
    if baseline not in wide or comparison not in wide:
        raise ValueError("Both comparison conditions are required")
    wide["difference"] = wide[baseline] - wide[comparison]
    paired = wide.reset_index()
    rows = []
    for names, block in paired.groupby(group_by, dropna=False):
        if not isinstance(names, tuple):
            names = (names,)
        # A missing repeat disqualifies that unit from the contrast; no survivor averaging.
        unit_values = block.groupby(unit)["difference"].agg(lambda x: x.mean() if np.isfinite(x).all() else np.nan)
        stats = paired_bootstrap(unit_values, seed=seed, draws=draws)
        stats["n_units"] = stats.pop("n_worlds")
        rows.append({**dict(zip(group_by, names)), "metric": metric, "contrast": f"{baseline} - {comparison}",
                     "n_units_total": len(unit_values), "n_missing_pairs": int(block["difference"].isna().sum()), **stats})
    return pd.DataFrame(rows)


def preview_plan(specs):
    return pd.DataFrame([{**{k: v for k, v in asdict(s).items() if k not in {"fit", "evaluation"}},
                          **{f"fit_{k}": v for k, v in asdict(s.fit).items()}} for s in specs])


def bundle_zip(bundle: RunBundle) -> bytes:
    """Prepare export in RAM. The caller must explicitly save/download these bytes."""
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("summary.csv", bundle.table.to_csv(index=False))
        archive.writestr("training_history.csv", bundle.history.to_csv(index=False))
        archive.writestr("subject_metrics.csv", bundle.subjects.to_csv(index=False))
        archive.writestr("plan.json", json.dumps(bundle.plans, indent=2))
        archive.writestr("environment.json", json.dumps(bundle.metadata, indent=2))
    return buffer.getvalue()


def persist_bundle(bundle: RunBundle, output_dir):
    """Explicit opt-in only. Does not infer a directory or trigger a browser download."""
    if output_dir is None or not str(output_dir).strip():
        raise ValueError("Choose an explicit output directory")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    # Content-addressed exports do not silently overwrite previous plan/results.
    payload = bundle_zip(bundle)
    name = f"results-{sha256(payload).hexdigest()[:16]}.zip"
    destination = path / name
    if not destination.exists():
        destination.write_bytes(payload)
    save_json(path / "latest-export.json", {"archive": name})
    return destination
