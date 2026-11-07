"""Finite stationary worlds with exactly auditable distributional controls.

State indices and observation tables are used by the simulator and evaluators only.
The neural training function receives observation pairs, never state labels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def circle_coordinates(n_states: int = 32, harmonic: int = 1) -> Array:
    if n_states < 8:
        raise ValueError("Use at least eight states.")
    theta = 2 * np.pi * np.arange(n_states) * harmonic / n_states
    return np.column_stack((np.cos(theta), np.sin(theta)))


def local_transition(n_states: int = 32, stay: float = 0.5) -> Array:
    """Lazy nearest-neighbor random walk on a cycle; uniform stationary measure."""
    circle_coordinates(n_states)
    if not 0 < stay < 1:
        raise ValueError("stay must be strictly between zero and one.")
    eye = np.eye(n_states)
    return stay * eye + (1 - stay) / 2 * (np.roll(eye, 1, 1) + np.roll(eye, -1, 1))


def transition_family(n_states: int = 32, seed: int = 0) -> tuple[dict[str, Array], NDArray]:
    local = local_transition(n_states)
    permutation = np.random.default_rng(seed).permutation(n_states)
    return {
        "local": local,
        "scrambled": local[np.ix_(permutation, permutation)],
        "iid": np.full_like(local, 1 / n_states),
    }, permutation


def validate_transition(transition: Array) -> None:
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("Transition must be a square matrix.")
    if not np.isfinite(transition).all() or np.min(transition) < -1e-12:
        raise ValueError("Transition contains invalid probabilities.")
    if not np.allclose(transition.sum(1), 1) or not np.allclose(transition, transition.T):
        raise ValueError("Expected a symmetric stochastic transition.")


def diagnostics(transition: Array, dim: int = 4) -> dict[str, float]:
    validate_transition(transition)
    n = len(transition)
    if not 1 <= dim < n - 1:
        raise ValueError("Embedding dimension must be between 1 and n_states - 2.")
    eigenvalues = np.linalg.eigvalsh(transition)[::-1]
    logp = np.zeros_like(transition)
    np.log(transition, out=logp, where=transition > 0)
    entropy = float(-(transition * logp).sum() / n)
    coords = circle_coordinates(n)
    distance2 = ((coords[:, None] - coords[None, :]) ** 2).sum(-1)
    return {
        "stationary_max_error": float(np.abs(transition.sum(0) / n - 1 / n).max()),
        "transition_entropy_nats": entropy,
        "lag1_latent_mi_nats": float(np.log(n) - entropy),
        "mixing_gap": float(1 - eigenvalues[1]),
        "embedding_gap": float(eigenvalues[dim] - eigenvalues[dim + 1]),
        "expected_chord_displacement2": float((transition * distance2).sum() / n),
    }


def spectral_embedding(transition: Array, dim: int = 4) -> tuple[Array, float]:
    """Hard-whitened temporal optimum, excluding the constant eigenfunction.

    A zero boundary gap means the selected subspace is not uniquely identified.
    This oracle uses the full transition matrix and is NOT a learned encoder.
    """
    audit = diagnostics(transition, dim)
    n = len(transition)
    # Remove the constant direction exactly, including for the degenerate iid kernel.
    centered = transition - np.ones((n, n)) / n
    _, basis = np.linalg.eigh(np.eye(n) - np.ones((n, n)) / n)
    basis = basis[:, 1:]
    values, vectors = np.linalg.eigh(basis.T @ centered @ basis)
    selected = vectors[:, np.argsort(values)[::-1][:dim]]
    return np.sqrt(n) * basis @ selected, audit["embedding_gap"]


def sensor_table(n_states: int, seed: int, aliased: bool = False, width: int = 12) -> Array:
    """Fixed nonlinear sensors. Aliased sensors identify antipodal states exactly."""
    if aliased and n_states % 2:
        raise ValueError("Antipodal aliasing requires an even state count.")
    count = n_states // 2 if aliased else n_states
    theta = 2 * np.pi * np.arange(count) / count
    basis = np.column_stack([np.cos(theta), np.sin(theta), np.cos(2 * theta), np.sin(2 * theta)])
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(4, width)) / 2
    table = np.tanh(basis @ weights + rng.normal(0, 0.2, size=width))
    table = (table - table.mean(0)) / np.maximum(table.std(0), 0.1)
    return np.tile(table, (2, 1)) if aliased else table


def image_table(n_states: int, side: int = 16) -> Array:
    """One smooth Gaussian marker on a circle, with shape (state, channel, y, x)."""
    if side < 8:
        raise ValueError("Use images at least 8 x 8.")
    grid = np.linspace(-1, 1, side)
    xx, yy = np.meshgrid(grid, grid)
    coordinates = 0.62 * circle_coordinates(n_states)
    distance2 = (xx[None] - coordinates[:, 0, None, None]) ** 2
    distance2 += (yy[None] - coordinates[:, 1, None, None]) ** 2
    return np.exp(-distance2 / (2 * 0.13**2))[:, None]


def sample_pairs(transition: Array, count: int, seed: int) -> NDArray[np.int64]:
    """Independent stationary length-two trajectories, not overlapping time windows."""
    validate_transition(transition)
    if count < 1:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(seed)
    n = len(transition)
    first = rng.integers(0, n, size=count)
    uniforms = rng.random(count)
    cumulative = np.cumsum(transition[first], axis=1)
    second = np.minimum((uniforms[:, None] > cumulative).sum(1), n - 1)
    return np.column_stack((first, second)).astype(np.int64)


def observe(table: Array, states: NDArray, noise: float, seed: int) -> NDArray[np.float32]:
    if noise < 0:
        raise ValueError("noise must be nonnegative.")
    clean = table[states]
    return (clean + np.random.default_rng(seed).normal(0, noise, clean.shape)).astype(np.float32)


def training_observations(
    transition: Array, table: Array, count: int, noise: float, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    pairs = sample_pairs(transition, count, seed)
    return observe(table, pairs[:, 0], noise, seed + 1), observe(table, pairs[:, 1], noise, seed + 2)


def observation_roughness(transition: Array, table: Array) -> float:
    flattened = table.reshape(len(table), -1)
    differences = ((flattened[:, None] - flattened[None, :]) ** 2).mean(-1)
    return float((transition * differences).sum() / len(table))
