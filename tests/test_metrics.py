import numpy as np
import pytest

from dynalign.metrics import effective_rank, linear_cka, paired_bootstrap, procrustes_test_error, rsa_spearman


def test_rotation_invariance_and_heldout_mapping():
    rng = np.random.default_rng(42)
    cal, test = rng.normal(size=(120, 4)), rng.normal(size=(120, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    assert linear_cka(test, 3 * test @ q + 2) == pytest.approx(1.0)
    assert rsa_spearman(test, 3 * test @ q + 2) == pytest.approx(1.0)
    assert procrustes_test_error(cal, cal @ q, test, test @ q) < 1e-20
    warped = test * np.array([1, 2, 5, 10])
    assert linear_cka(test, warped) < 0.9
    assert procrustes_test_error(cal, cal, test, rng.normal(size=test.shape)) > 0.5


def test_collapse_is_not_perfect_alignment():
    flat = np.ones((30, 4))
    assert np.isnan(linear_cka(flat, flat))
    assert np.isnan(procrustes_test_error(flat, flat, flat, flat))
    assert effective_rank(flat) == 0


def test_bootstrap_requires_independent_world_count():
    assert np.isnan(paired_bootstrap([0.1, 0.2, 0.3])["ci_low"])
    result = paired_bootstrap([0.1] * 5)
    assert result["n_worlds"] == 5
    assert result["ci_low"] == pytest.approx(0.1)
