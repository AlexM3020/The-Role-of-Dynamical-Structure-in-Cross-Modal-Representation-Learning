"""Bounded experiment grids, independent seeds, evaluation, and resumable artifacts."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess

import numpy as np
import pandas as pd
import torch

from . import __version__
from .metrics import effective_rank, linear_cka, procrustes_test_error, ridge_test_r2, rsa_spearman
from .training import TrainConfig, build_encoder, encode, train_encoder
from .worlds import (
    circle_coordinates, diagnostics, image_table, observe, observation_roughness,
    sensor_table, spectral_embedding, training_observations, transition_family,
)


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def environment() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "package": __version__, "source_fingerprint": source_fingerprint(), "git_commit": commit,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "torch")},
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def grid(experiment: str, profile: str = "smoke") -> list[dict]:
    """Smoke runs check execution; pilot/full increase replication, not truthfulness."""
    if profile not in {"smoke", "pilot", "full"}:
        raise ValueError("profile must be smoke, pilot, or full")
    if experiment not in {"sample_efficiency", "robustness", "pixels"}:
        raise ValueError("Unknown experiment")
    worlds = {"smoke": 1, "pilot": 3, "full": 5}[profile]
    repeats = 2 if profile == "full" else 1
    steps = {"smoke": 40, "pilot": 300, "full": 800}[profile]
    budgets = {"smoke": [256], "pilot": [512, 2048], "full": [512, 2048, 8192]}[profile]
    cases = [("clean", 0.05, False)]
    conditions = ["local", "scrambled", "iid"] if experiment == "sample_efficiency" else ["local", "scrambled"]
    if experiment == "robustness":
        budgets = [256 if profile == "smoke" else 2048]
        cases = [("clean", 0.05, False), ("noisy", 0.3, False), ("aliased_b", 0.05, True)]
    rows = []
    for world in range(worlds):
        for repeat in range(repeats):
            for count in budgets:
                for case, noise, alias in cases:
                    for condition in conditions:
                        rows.append({
                            "experiment": experiment, "profile": profile, "world": world,
                            "repeat": repeat, "n_pairs": count, "condition": condition,
                            "case": case, "noise": noise, "aliased_b": alias,
                            "pixels": experiment == "pixels", "n_states": 32,
                            "steps": steps, "width": 64, "dim": 4,
                        })
    return rows


def _tables(spec: dict):
    n, world = spec["n_states"], spec["world"]
    a = image_table(n) if spec["pixels"] else sensor_table(n, 1000 + world * 10)
    b = sensor_table(n, 1001 + world * 10, aliased=spec["aliased_b"])
    return a, b


def _evaluate(models, tables, spec, transition, device) -> tuple[dict, dict]:
    n = spec["n_states"]
    # Two independent evaluation draws at every state; no training examples reused.
    cal_states = np.tile(np.arange(n), 8)
    test_states = np.tile(np.arange(n), 16)
    embeddings = []
    base = 800_000 + 10_000 * spec["world"] + 100 * spec["repeat"]
    for m, (model, table) in enumerate(zip(models, tables)):
        noise = 0.03 if spec["pixels"] and m == 0 else spec["noise"]
        cal = encode(model, observe(table, cal_states, noise, base + 10 * m), device)
        test = encode(model, observe(table, test_states, noise, base + 10 * m + 1), device)
        embeddings.append((cal, test))
    (ca, ta), (cb, tb) = embeddings
    centroids_a = ta.reshape(16, n, -1).mean(0)
    centroids_b = tb.reshape(16, n, -1).mean(0)
    target = circle_coordinates(n)
    quotient = circle_coordinates(n, harmonic=2)
    oracle, gap = spectral_embedding(transition, spec["dim"])
    shuffled = np.random.default_rng(base + 50).permutation(len(tb))
    result = {
        "cka": linear_cka(ta, tb),
        "centroid_cka": linear_cka(centroids_a, centroids_b),
        "centroid_rsa": rsa_spearman(centroids_a, centroids_b),
        "shuffled_cka": linear_cka(ta, tb[shuffled]),
        "procrustes_error": procrustes_test_error(ca, cb, ta, tb),
    }
    for label, (cal, test) in zip(("a", "b"), embeddings):
        variance = float(np.mean(np.var(test, axis=0)))
        rank = effective_rank(test)
        result[f"variance_{label}"] = variance
        result[f"effective_rank_{label}"] = rank
        result[f"collapse_{label}"] = bool(variance < 1e-6 or rank < 0.5 * spec["dim"])
        result[f"coordinate_r2_{label}"] = ridge_test_r2(cal, target[cal_states], test, target[test_states])
        result[f"quotient_r2_{label}"] = ridge_test_r2(cal, quotient[cal_states], test, quotient[test_states])
        # This is transition-specific and is not the fixed circular-coordinate probe.
        result[f"spectral_r2_{label}"] = ridge_test_r2(cal, oracle[cal_states], test, oracle[test_states]) if gap > 1e-8 else float("nan")
        expected_next = transition @ target
        result[f"next_mean_r2_{label}"] = ridge_test_r2(cal, expected_next[cal_states], test, expected_next[test_states])
    result["any_collapse"] = result["collapse_a"] or result["collapse_b"]
    arrays = {"cal_a": ca, "cal_b": cb, "test_a": ta, "test_b": tb,
              "cal_states": cal_states, "test_states": test_states}
    return result, arrays


def run_condition(spec: dict, directory: Path, device: str = "cpu") -> dict:
    family, permutation = transition_family(spec["n_states"], 500 + spec["world"])
    transition = family[spec["condition"]]
    tables = _tables(spec)
    train_config = TrainConfig(steps=spec["steps"], width=spec["width"], dim=spec["dim"])
    base = 20_000 + spec["world"] * 10_000 + spec["repeat"] * 1000
    models = [build_encoder(tuple(table.shape[1:]), train_config, base + 100 * m).to(device)
              for m, table in enumerate(tables)]
    initial, _ = _evaluate(models, tables, spec, transition, device)
    training = []
    for m, (model, table) in enumerate(zip(models, tables)):
        # No paired training sequence is shared between modalities.
        data_seed = base + 100 * m + 10
        noise = 0.03 if spec["pixels"] and m == 0 else spec["noise"]
        first, second = training_observations(transition, table, spec["n_pairs"], noise, data_seed)
        training.append(train_encoder(model, first, second, train_config, base + 100 * m + 20,
                                      device, directory / f"encoder_{m}.pt"))
    metrics, arrays = _evaluate(models, tables, spec, transition, device)
    arrays.update({"transition": transition, "permutation": permutation,
                   "table_a": tables[0], "table_b": tables[1]})
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "embeddings.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, directory / "embeddings.npz")
    row = {**spec, **metrics, "initial_cka": initial["cka"],
           "cka_gain": metrics["cka"] - initial["cka"],
           "training_seconds": sum(item["training_seconds"] for item in training),
           "parameters_a": sum(p.numel() for p in models[0].parameters()),
           "parameters_b": sum(p.numel() for p in models[1].parameters()),
           **diagnostics(transition, spec["dim"]),
           "sensor_roughness_a": observation_roughness(transition, tables[0]),
           "sensor_roughness_b": observation_roughness(transition, tables[1])}
    save_json(directory / "result.json", {"row": row, "training": training,
                                          "config": asdict(train_config), "environment": environment()})
    return row


def run_grid(specs: list[dict], output_dir: str | Path, device: str = "auto") -> pd.DataFrame:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use device='cpu' or 'auto'.")
    # Small models are faster and more predictable with modest CPU thread counts.
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = environment()
    save_json(output_dir / "environment.json", manifest)
    rows = []
    for index, spec in enumerate(specs):
        identity = {"spec": spec, "source": manifest["source_fingerprint"],
                    "device": device, "packages": manifest["packages"]}
        run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        directory = output_dir / run_id
        result_file = directory / "result.json"
        if result_file.exists():
            row = json.loads(result_file.read_text(encoding="utf-8"))["row"]
            print(f"[{index + 1}/{len(specs)}] cached {spec['condition']} {spec['case']} N={spec['n_pairs']}")
        else:
            save_json(directory / "run.json", identity)
            print(f"[{index + 1}/{len(specs)}] {spec['condition']} {spec['case']} N={spec['n_pairs']} world={spec['world']} repeat={spec['repeat']}", flush=True)
            row = run_condition(spec, directory, device)
        rows.append({**row, "run_id": run_id})
        temporary = output_dir / "summary.tmp.csv"
        pd.DataFrame(rows).to_csv(temporary, index=False)
        os.replace(temporary, output_dir / "summary.csv")
    return pd.DataFrame(rows)
