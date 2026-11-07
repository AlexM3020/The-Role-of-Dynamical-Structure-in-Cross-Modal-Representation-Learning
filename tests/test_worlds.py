import numpy as np
import pytest

from dynalign.worlds import (
    circle_coordinates, diagnostics, image_table, local_transition, observe,
    sample_pairs, sensor_table, spectral_embedding, transition_family,
)


def test_main_controls_are_exact():
    family, permutation = transition_family(32, 41)
    local, scrambled = family["local"], family["scrambled"]
    np.testing.assert_allclose(np.linalg.eigvalsh(local), np.linalg.eigvalsh(scrambled), atol=1e-12)
    for key in ("transition_entropy_nats", "lag1_latent_mi_nats", "mixing_gap", "embedding_gap"):
        assert diagnostics(local)[key] == pytest.approx(diagnostics(scrambled)[key], abs=1e-12)
    assert diagnostics(local)["expected_chord_displacement2"] < diagnostics(scrambled)["expected_chord_displacement2"]
    assert diagnostics(local)["stationary_max_error"] < 1e-12
    table = sensor_table(32, 10)
    pairs = sample_pairs(local, 512, 15)
    inverse = np.argsort(permutation)
    np.testing.assert_allclose(table[pairs], table[permutation][inverse[pairs]], atol=1e-12)


def test_embedding_boundary_gap_and_whitening():
    p = local_transition(32)
    h, gap = spectral_embedding(p, 4)
    assert gap > 0
    np.testing.assert_allclose(h.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(h.T @ h / 32, np.eye(4), atol=1e-12)
    assert diagnostics(p, 3)["embedding_gap"] < 1e-12
    iid = np.ones((32, 32)) / 32
    assert spectral_embedding(iid)[1] < 1e-12


def test_exact_aliasing_and_independent_measurement_noise():
    table = sensor_table(32, 12, aliased=True)
    np.testing.assert_array_equal(table[:16], table[16:])
    assert not np.allclose(circle_coordinates(32)[:16], circle_coordinates(32)[16:])
    np.testing.assert_allclose(circle_coordinates(32, 2)[:16], circle_coordinates(32, 2)[16:], atol=1e-12)
    states = np.arange(32)
    assert not np.array_equal(observe(table, states, 0.1, 1), observe(table, states, 0.1, 2))
    assert image_table(32).shape == (32, 1, 16, 16)


def test_transition_sampling():
    pairs = sample_pairs(local_transition(16), 40000, 13)
    assert set(((pairs[:, 1] - pairs[:, 0]) % 16).tolist()) == {0, 1, 15}
    frequencies = np.bincount(pairs[:, 0], minlength=16) / len(pairs)
    np.testing.assert_allclose(frequencies, 1 / 16, atol=0.005)
