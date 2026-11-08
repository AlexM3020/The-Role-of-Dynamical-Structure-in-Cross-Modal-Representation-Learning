"""Raw HAPT accelerometer/gyroscope experiment; loading and results are RAM-first.

Official source: https://archive.ics.uci.edu/dataset/341/
No dataset is fetched at import, and no archive is extracted to disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch

from .experiments import environment
from .metrics import linear_cka, procrustes_test_error
from .revised import (
    EvalSettings, FitSettings, RunBundle, derived_seed, embeddings, fit_encoder,
    make_encoder, persist_bundle, rank_diagnostics, select_device,
)

HAPT_URL = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"


@dataclass(frozen=True)
class WindowSettings:
    length: int = 128
    stride: int = 128
    label_purity: float = 0.8
    max_windows_per_record: int | None = None

    def __post_init__(self):
        if self.length < 8 or self.stride < self.length:
            raise ValueError("Use non-overlapping windows: stride >= length >= 8")
        if not 0 < self.label_purity <= 1:
            raise ValueError("label_purity must be in (0, 1]")
        if self.max_windows_per_record is not None and self.max_windows_per_record < 2:
            raise ValueError("Need >=2 windows per recording")


@dataclass
class SensorWindows:
    acceleration: np.ndarray
    gyroscope: np.ndarray
    subject: np.ndarray
    recording: np.ndarray
    start: np.ndarray
    label: np.ndarray
    provenance: dict = field(default_factory=dict)

    def take(self, mask):
        return SensorWindows(*(getattr(self, name)[mask] for name in (
            "acceleration", "gyroscope", "subject", "recording", "start", "label"
        )), provenance=self.provenance.copy())


def load_hapt(*, archive_path=None, allow_download=False, window=None, max_bytes=600_000_000):
    """Use a supplied archive or explicitly opt into a bounded in-memory download."""
    if window is None:
        window = WindowSettings()
    if archive_path is None:
        if not allow_download:
            raise ValueError("Download disabled. Set allow_download=True or supply archive_path.")
        with urlopen(HAPT_URL, timeout=120) as response:
            data = response.read(max_bytes + 1)
        source = HAPT_URL
    else:
        path = Path(archive_path)
        if path.stat().st_size > max_bytes:
            raise ValueError("Archive exceeds the configured memory/download limit")
        data = path.read_bytes()
        source = path.name
    if len(data) > max_bytes:
        raise ValueError("Archive exceeds the configured memory/download limit")
    result = parse_hapt_zip(data, window)
    result.provenance.update({"source": source, "archive_sha256": sha256(data).hexdigest(),
                              "window_settings": asdict(window), "license": "CC BY 4.0",
                              "citation": "Reyes-Ortiz et al. (2015), UCI HAPT, doi:10.24432/C54G7M"})
    return result


def parse_hapt_zip(data: bytes, window: WindowSettings):
    with ZipFile(BytesIO(data)) as archive:
        names = archive.namelist()
        acceleration_names = [name for name in names if re.search(r"(?:^|/)RawData/acc_exp\d+_user\d+\.txt$", name, re.I)]
        # Some UCI wrappers contain the original dataset archive as a member.
        if not acceleration_names:
            nested = [name for name in names if name.lower().endswith(".zip")]
            if len(nested) == 1:
                return parse_hapt_zip(archive.read(nested[0]), window)
            raise ValueError("No raw HAPT accelerometer files found; 561-feature tables are not raw signals")
        labels_files = [name for name in names if name.lower().endswith("rawdata/labels.txt")]
        if len(labels_files) != 1:
            raise ValueError("Expected one RawData/labels.txt")
        labels = np.atleast_2d(np.loadtxt(BytesIO(archive.read(labels_files[0])), dtype=int))
        if labels.shape[1] != 5:
            raise ValueError("Expected recording, subject, activity, start, end label columns")
        a_windows, b_windows, subjects, recordings, starts, classes = [], [], [], [], [], []
        for name in sorted(acceleration_names):
            match = re.search(r"acc_exp(\d+)_user(\d+)\.txt$", name, re.I)
            recording, subject = map(int, match.groups())
            gyro_name = re.sub(r"acc_exp", "gyro_exp", name)
            a = np.loadtxt(BytesIO(archive.read(name)), dtype=np.float32)
            b = np.loadtxt(BytesIO(archive.read(gyro_name)), dtype=np.float32)
            if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
                raise ValueError(f"Paired signal shape mismatch for recording {recording}")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Non-finite raw measurements")
            activity = np.zeros(len(a), dtype=int)
            for _, _, label, first, last in labels[(labels[:, 0] == recording) & (labels[:, 1] == subject)]:
                # HAPT label bounds are 1-based inclusive sample indices.
                if not 1 <= first <= last <= len(a):
                    raise ValueError("Activity bounds are outside the raw recording")
                activity[first - 1:last] = label
            count = 0
            for start in range(0, len(a) - window.length + 1, window.stride):
                counts = np.bincount(activity[start:start + window.length], minlength=13)
                label = int(np.argmax(counts))
                if counts[label] / window.length < window.label_purity:
                    label = 0
                a_windows.append(a[start:start + window.length].T)
                b_windows.append(b[start:start + window.length].T)
                subjects.append(subject)
                recordings.append(recording)
                starts.append(start)
                classes.append(label)
                count += 1
                if window.max_windows_per_record is not None and count >= window.max_windows_per_record:
                    break
        if not a_windows:
            raise ValueError("No complete windows available")
        return SensorWindows(np.stack(a_windows), np.stack(b_windows), np.array(subjects),
                             np.array(recordings), np.array(starts), np.array(classes))


def split_and_standardize(data, train_subjects, calibration_subjects, test_subjects):
    groups = [set(train_subjects), set(calibration_subjects), set(test_subjects)]
    if not all(groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Training, calibration and test participants must be nonempty and disjoint")
    available = set(data.subject.tolist())
    if not set.union(*groups).issubset(available):
        raise ValueError("A requested participant is not present in the loaded archive")
    parts = [data.take(np.isin(data.subject, list(ids))) for ids in groups]
    normalization = {}
    for channel in ("acceleration", "gyroscope"):
        training = getattr(parts[0], channel)
        mean = training.mean(axis=(0, 2), keepdims=True)
        scale = np.maximum(training.std(axis=(0, 2), keepdims=True), 1e-6)
        normalization[channel] = {"mean": mean.ravel().tolist(), "std": scale.ravel().tolist()}
        for part in parts:
            setattr(part, channel, ((getattr(part, channel) - mean) / scale).astype(np.float32))
    return (*parts, normalization)


def temporal_pairs(recordings, count, condition, permutation_seed, sample_seed):
    """Reflecting nearest-neighbor kernel, permuted consistently within recordings.

    P = .5 I + .25 left + .25 right (out-of-bound steps stay at the endpoint).
    This kernel is symmetric and uniform-stationary. Scrambling is P[pi, pi].
    No activity labels enter pair selection, and recordings are never bridged.
    """
    if condition not in {"local", "scrambled", "iid"}:
        raise ValueError("Unknown temporal condition")
    permutation_rng = np.random.default_rng(permutation_seed)
    left, right = np.arange(len(recordings)), np.arange(len(recordings))
    members = {}
    for recording in np.unique(recordings):
        ids = np.flatnonzero(recordings == recording)
        if len(ids) < 2:
            raise ValueError("Every training recording needs at least two windows")
        permutation = permutation_rng.permutation(len(ids)) if condition == "scrambled" else np.arange(len(ids))
        inverse = np.argsort(permutation)
        left[ids] = ids[inverse[np.maximum(permutation - 1, 0)]]
        right[ids] = ids[inverse[np.minimum(permutation + 1, len(ids) - 1)]]
        members[recording] = ids
    rng = np.random.default_rng(sample_seed)
    first = rng.integers(len(recordings), size=count)
    second = first.copy()
    draws = rng.random(count)
    if condition == "iid":
        for recording, ids in members.items():
            mask = recordings[first] == recording
            second[mask] = rng.choice(ids, mask.sum())
    else:
        mask_left, mask_right = draws < 0.25, draws >= 0.75
        second[mask_left] = left[first[mask_left]]
        second[mask_right] = right[first[mask_right]]
    return np.column_stack((first, second))


def activity_probe(train_h, train_labels, test_h, test_labels, ridge, valid_labels):
    train_mask = np.isin(train_labels, valid_labels)
    test_mask = np.isin(test_labels, valid_labels)
    classes = np.array(valid_labels)
    if not test_mask.any() or not all(np.any(train_labels[train_mask] == c) for c in classes):
        return {"balanced_accuracy": np.nan, "macro_f1": np.nan, "probe_n": int(test_mask.sum())}
    x, y = train_h[train_mask], train_labels[train_mask]
    mx = x.mean(0)
    scale = np.maximum(x.std(0), 1e-6)
    design = np.column_stack(((x - mx) / scale, np.ones(len(x))))
    onehot = (y[:, None] == classes[None]).astype(float)
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ onehot)
    test_design = np.column_stack(((test_h[test_mask] - mx) / scale, np.ones(test_mask.sum())))
    predicted = classes[np.argmax(test_design @ weights, 1)]
    truth = test_labels[test_mask]
    recalls, f1s = [], []
    for c in classes:
        tp = np.sum((predicted == c) & (truth == c))
        fn = np.sum((predicted != c) & (truth == c))
        fp = np.sum((predicted == c) & (truth != c))
        if tp + fn:
            recalls.append(tp / (tp + fn))
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)),
            "probe_n": int(test_mask.sum())}


@dataclass(frozen=True)
class RealSpec:
    repeat: int = 0
    master_seed: int = 20261108
    condition: str = "local"
    regime: str = "baseline"
    architecture: str = "temporal_cnn"
    width: int = 16
    dim: int = 8
    n_pairs: int = 512
    valid_activity_labels: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    minimum_group_size: int = 32
    fit: FitSettings = field(default_factory=FitSettings)
    evaluation: EvalSettings = field(default_factory=EvalSettings)

    def __post_init__(self):
        if self.condition not in {"local", "scrambled", "iid"} or self.n_pairs < 2:
            raise ValueError("Invalid real-data transition settings")
        if self.architecture not in {"linear", "temporal_cnn", "order_invariant"}:
            raise ValueError("Unknown real-data encoder")
        if min(self.width, self.dim, self.minimum_group_size) < 1:
            raise ValueError("Positive dimensions and group sizes required")


def run_real(specs, train, calibration, test, *, normalization, device="auto", threads=2, output_dir=None):
    if set(train.subject) & set(calibration.subject) or set(train.subject) & set(test.subject) or set(calibration.subject) & set(test.subject):
        raise ValueError("Participant leakage between splits")
    device = select_device(device, threads)
    metadata = {**environment(), "engine": "hapt-memory-first", "device": device, "threads": threads,
                "dataset": train.provenance, "normalization": normalization,
                "subjects": {"train": sorted(set(train.subject.tolist())),
                             "calibration": sorted(set(calibration.subject.tolist())),
                             "test": sorted(set(test.subject.tolist()))}}
    # Include window membership and values so changed data cannot reuse old checkpoints.
    hasher = sha256()
    for split in (train, calibration, test):
        for name in ("acceleration", "gyroscope", "subject", "recording", "start", "label"):
            hasher.update(np.ascontiguousarray(getattr(split, name)).tobytes())
    metadata["prepared_data_sha256"] = hasher.hexdigest()
    bundle = RunBundle(metadata=metadata)
    for index, spec in enumerate(specs):
        plan = asdict(spec)
        run_id = sha256(json.dumps({"spec": plan, "environment": metadata}, sort_keys=True).encode()).hexdigest()[:20]
        bundle.plans.append({"run_id": run_id, **plan})
        print(f"[{index + 1}/{len(specs)}] real sensors: {spec.regime}, {spec.condition}, seed={spec.repeat}", flush=True)
        snapshots, elapsed, parameters = [], 0.0, []
        for m, channel in enumerate(("acceleration", "gyroscope")):
            x = getattr(train, channel)
            pair_ids = temporal_pairs(train.recording, spec.n_pairs, spec.condition,
                                      derived_seed(spec.master_seed, 0, spec.repeat, 1),
                                      derived_seed(spec.master_seed, 0, spec.repeat, 10 + m))
            model = make_encoder(x.shape[1:], spec.architecture, spec.width, spec.dim,
                                 derived_seed(spec.master_seed, 0, spec.repeat, 20 + m)).to(device)
            parameters.append(sum(p.numel() for p in model.parameters()))
            def evaluate(current, channel=channel):
                return {name: torch.from_numpy(embeddings(current, getattr(split, channel), device, spec.evaluation.batch_size))
                        for name, split in (("train", train), ("cal", calibration), ("test", test))}
            checkpoint = None if output_dir is None else Path(output_dir) / run_id / f"encoder_{m}.pt"
            snapshot, history, seconds = fit_encoder(model, (x[pair_ids[:, 0]], x[pair_ids[:, 1]]), spec.fit,
                                                     derived_seed(spec.master_seed, 0, spec.repeat, 30 + m),
                                                     device, evaluate, checkpoint)
            snapshots.append(snapshot)
            elapsed += seconds
            bundle.histories.extend({"run_id": run_id, "modality": m, **item} for item in history)
        for step in sorted(snapshots[0]):
            a, b = [{name: value.cpu().numpy() for name, value in side[step].items()} for side in snapshots]
            common = {k: v for k, v in plan.items() if k not in {"fit", "evaluation"}}
            common.update({"run_id": run_id, "step": step})
            global_row = {**common, "cka": linear_cka(a["test"], b["test"]),
                          "procrustes_error": procrustes_test_error(a["cal"], b["cal"], a["test"], b["test"]),
                          "run_seconds_including_evaluation": elapsed,
                          "parameters_a": parameters[0], "parameters_b": parameters[1], "test_windows": len(test.subject)}
            for suffix, side in (("a", a), ("b", b)):
                global_row.update({f"{key}_{suffix}": value for key, value in rank_diagnostics(side["test"], spec.evaluation).items()})
            bundle.rows.append(global_row)
            for subject in sorted(set(test.subject)):
                mask = test.subject == subject
                row = {**common, "subject": int(subject), "n_windows": int(mask.sum()),
                       "cka": linear_cka(a["test"][mask], b["test"][mask]) if mask.sum() >= spec.minimum_group_size else np.nan,
                       "procrustes_error": procrustes_test_error(a["cal"], b["cal"], a["test"][mask], b["test"][mask])}
                conditional = []
                for activity in spec.valid_activity_labels:
                    restricted = mask & (test.label == activity)
                    if restricted.sum() >= spec.minimum_group_size:
                        conditional.append(linear_cka(a["test"][restricted], b["test"][restricted]))
                row["within_activity_cka"] = float(np.mean(conditional)) if conditional else np.nan
                row["n_valid_activity_groups"] = len(conditional)
                for suffix, side in (("a", a), ("b", b)):
                    row.update({f"{key}_{suffix}": value for key, value in activity_probe(
                        side["train"], train.label, side["test"][mask], test.label[mask],
                        spec.evaluation.ridge, spec.valid_activity_labels).items()})
                bundle.subject_rows.append(row)
        if output_dir is not None:
            persist_bundle(bundle, output_dir)
    return bundle
