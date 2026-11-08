from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch

from dynalign.revised import (
    EvalSettings, FitSettings, SyntheticSpec, attainable_reference, bundle_zip,
    make_encoder, paired_contrasts, posterior_target, run_synthetic, temporal_loss,
    state_noise_diagnostics,
)
from dynalign.hapt import (
    RealSpec, WindowSettings, load_hapt, parse_hapt_zip,
    run_real, split_and_standardize, temporal_pairs,
)
from dynalign.worlds import circle_coordinates, sensor_table


def tiny_fit(**changes):
    return replace(FitSettings(steps=3, checkpoints=(1, 3), batch_size=16, log_every=1), **changes)


def test_memory_default_does_not_create_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    specifications = [SyntheticSpec(condition=c, n_pairs=48, dim=2, width=8,
                                   fit=tiny_fit(), evaluation=EvalSettings(2, 2))
                      for c in ("local", "scrambled")]
    bundle = run_synthetic(specifications, device="cpu", threads=1)
    assert set(bundle.table.step) == {0, 1, 3}
    assert len(bundle.table) == 6
    assert not list(tmp_path.iterdir())
    assert bundle.table.cka.notna().all()
    with ZipFile(BytesIO(bundle_zip(bundle))) as archive:
        plans = json.loads(archive.read("plan.json"))
        assert plans[0]["fit"]["invariance_weight"] == 25
        assert "summary.csv" in archive.namelist()
    assert not list(tmp_path.iterdir())  # Preparing an export is still memory-only.


@pytest.mark.parametrize("objective", ["vicreg", "cov_identity"])
def test_weights_reach_actual_loss(objective):
    a = torch.tensor([[1., 0.], [0., 2.], [2., 1.]])
    b = a.flip(0)
    cfg = tiny_fit(objective=objective, invariance_weight=2, variance_weight=3,
                   covariance_weight=4, identity_weight=7)
    loss, pieces = temporal_loss(a, b, cfg)
    expected = 2 * pieces["invariance"] + (3 * pieces["variance"] + 4 * pieces["covariance"]
                                          if objective == "vicreg" else 7 * pieces["covariance_identity"])
    torch.testing.assert_close(loss, expected)


def test_aliasing_reference_and_zero_noise():
    table = sensor_table(32, 3, aliased=True)
    target = circle_coordinates(32)
    mean = posterior_target(table, table, target, 0)
    np.testing.assert_allclose(mean, 0, atol=1e-10)
    quotient = circle_coordinates(32, 2)
    np.testing.assert_allclose(posterior_target(table, table, quotient, 0), quotient, atol=1e-10)
    reference = attainable_reference(table, 0, target, 2, 4)
    assert reference["bayes_mc_r2"] == pytest.approx(0, abs=1e-10)


def test_state_noise_requires_balanced_repeated_measurements():
    with pytest.raises(ValueError, match="balanced"):
        state_noise_diagnostics(np.ones((100, 4)), 32)
    with pytest.raises(ValueError, match="balanced"):
        state_noise_diagnostics(np.ones((32, 4)), 32)


def test_paired_statistics_do_not_count_repeats_as_worlds():
    rows = [{"world": w, "repeat": r, "regime": "base", "condition": c,
             "n_pairs": 1, "step": 3, "cka": 0.8 if c == "local" else 0.4}
            for w in range(5) for r in range(2) for c in ("local", "scrambled")]
    table = pd.DataFrame(rows)
    result = paired_contrasts(table)
    assert result.iloc[0].n_units == 5
    assert result.iloc[0]["mean"] == pytest.approx(0.4)
    table.loc[0, "cka"] = np.nan
    result = paired_contrasts(table)
    assert result.iloc[0].n_units == 4
    assert result.iloc[0].n_missing_pairs == 1
    assert np.isnan(result.iloc[0].ci_low)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_contrasts(pd.concat([table, table]))


def test_loss_and_optimizer_changes_change_export_identity():
    a = SyntheticSpec(fit=tiny_fit())
    b = replace(a, fit=replace(a.fit, covariance_weight=10))
    assert a != b
    assert make_encoder((12,), "linear", 64, 4, 1)[1].out_features == 4
    with pytest.raises(ValueError):
        FitSettings(steps=2, checkpoints=(3,))


def fixture_archive():
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        labels = []
        for user in range(1, 7):
            rng = np.random.default_rng(user)
            for sensor in ("acc", "gyro"):
                values = rng.normal(size=(64, 3))
                text = "\n".join(" ".join(map(str, row)) for row in values)
                archive.writestr(f"HAPT/RawData/{sensor}_exp{user:02d}_user{user:02d}.txt", text)
            labels.append(f"{user} {user} 1 1 32\n{user} {user} 2 33 64")
        archive.writestr("HAPT/RawData/labels.txt", "\n".join(labels))
    return payload.getvalue()


def test_real_archive_parsing_and_no_implicit_download():
    with pytest.raises(ValueError, match="Download disabled"):
        load_hapt()
    with pytest.raises(ValueError, match="non-overlapping"):
        WindowSettings(length=128, stride=64)
    data = parse_hapt_zip(fixture_archive(), WindowSettings(length=16, stride=16))
    assert data.acceleration.shape == (24, 3, 16)
    np.testing.assert_array_equal(data.label[:4], [1, 1, 2, 2])
    train, cal, test, normalization = split_and_standardize(data, [1, 2], [3, 4], [5, 6])
    np.testing.assert_allclose(train.acceleration.mean(axis=(0, 2)), 0, atol=1e-6)
    assert normalization["acceleration"]["std"]
    assert not set(train.subject) & set(test.subject)
    with pytest.raises(ValueError, match="disjoint"):
        split_and_standardize(data, [1, 2], [2, 3], [4, 5])


def test_real_pair_graph_preserves_recordings_and_uniform_marginals():
    records = np.repeat([1, 2], 8)
    for condition in ("local", "scrambled"):
        pairs = temporal_pairs(records, 80000, condition, 13, 14)
        np.testing.assert_array_equal(records[pairs[:, 0]], records[pairs[:, 1]])
        marginal = np.bincount(pairs[:, 1], minlength=16) / len(pairs)
        np.testing.assert_allclose(marginal, 1 / 16, atol=0.005)


def test_real_workflow_on_explicit_test_fixture_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = parse_hapt_zip(fixture_archive(), WindowSettings(length=16, stride=16))
    data.provenance["test_fixture"] = True  # Never label fixture output as real HAPT evidence.
    train, cal, test, norm = split_and_standardize(data, [1, 2], [3, 4], [5, 6])
    specs = [RealSpec(condition=c, n_pairs=32, width=4, dim=2, fit=tiny_fit(),
                      minimum_group_size=2, valid_activity_labels=(1, 2)) for c in ("local", "scrambled")]
    bundle = run_real(specs, train, cal, test, normalization=norm, device="cpu", threads=1)
    assert len(bundle.rows) == 6
    assert len(bundle.subject_rows) == 12
    assert bundle.metadata["dataset"]["test_fixture"]
    assert not list(tmp_path.iterdir())


def test_opt_in_persistence_and_resume(tmp_path):
    spec = SyntheticSpec(n_pairs=32, dim=2, width=8, fit=tiny_fit(), evaluation=EvalSettings(2, 2))
    first = run_synthetic([spec], output_dir=tmp_path, device="cpu", threads=1)
    second = run_synthetic([spec], output_dir=tmp_path, device="cpu", threads=1)
    np.testing.assert_array_equal(first.table.cka.values, second.table.cka.values)
    assert list(tmp_path.glob("*/encoder_0.pt"))
    assert list(tmp_path.glob("results-*.zip"))
