"""Held-out geometry metrics with explicit handling of collapsed representations."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def centered(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(0, keepdims=True)


def linear_cka(x, y) -> float:
    x, y = centered(x), centered(y)
    numerator = np.linalg.norm(x.T @ y, "fro") ** 2
    denominator = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(numerator / denominator) if denominator > 1e-12 else float("nan")


def effective_rank(x) -> float:
    """Entropy effective rank of covariance eigenvalues; not an MDL estimate."""
    values = np.linalg.svd(centered(x), compute_uv=False) ** 2
    if values.sum() < 1e-12:
        return 0.0
    probabilities = values[values > 1e-12] / values.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def rsa_spearman(x, y) -> float:
    """Spearman correlation of upper-triangle squared Euclidean distances."""
    x, y = centered(x), centered(y)
    indices = np.triu_indices(len(x), 1)
    dx = ((x[:, None] - x[None, :]) ** 2).sum(-1)[indices]
    dy = ((y[:, None] - y[None, :]) ** 2).sum(-1)[indices]
    rx, ry = rankdata(dx), rankdata(dy)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def procrustes_test_error(cal_x, cal_y, test_x, test_y) -> float:
    """Fit mean, one global scale per modality, and orthogonal map on calibration only.

    No per-coordinate whitening: arbitrary linear distortions remain detectable.
    Error is normalized by target test energy; zero is a perfect held-out fit.
    """
    mx, my = np.mean(cal_x, 0), np.mean(cal_y, 0)
    sx = np.sqrt(np.mean((cal_x - mx) ** 2))
    sy = np.sqrt(np.mean((cal_y - my) ** 2))
    if sx < 1e-8 or sy < 1e-8:
        return float("nan")
    x, y = (cal_x - mx) / sx, (cal_y - my) / sy
    if np.linalg.matrix_rank(x) < x.shape[1] or np.linalg.matrix_rank(y) < y.shape[1]:
        return float("nan")
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    prediction = (test_x - mx) / sx @ (u @ vt)
    target = (test_y - my) / sy
    denominator = float(np.mean(target**2))
    return float(np.mean((prediction - target) ** 2) / denominator) if denominator > 1e-12 else float("nan")


def ridge_test_r2(cal_x, cal_target, test_x, test_target, ridge: float = 1e-3) -> float:
    """A fixed-complexity linear probe. It does not assert causal identification."""
    mx, my = np.mean(cal_x, 0), np.mean(cal_target, 0)
    x = np.asarray(cal_x) - mx
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ (cal_target - my))
    predictions = (test_x - mx) @ weights + my
    denominator = np.sum((test_target - my) ** 2)
    return float(1 - np.sum((predictions - test_target) ** 2) / denominator) if denominator > 1e-12 else float("nan")


def paired_bootstrap(values, seed: int = 0, draws: int = 5000) -> dict[str, float]:
    """Bootstrap independent world-level paired differences, not individual frames."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n_worlds": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    result = {"n_worlds": int(len(values)), "mean": float(values.mean())}
    if len(values) < 5:
        return {**result, "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (draws, len(values)), replace=True).mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {**result, "ci_low": float(low), "ci_high": float(high)}
