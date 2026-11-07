from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from dynalign.experiments import grid, run_grid
from dynalign.training import TrainConfig, build_encoder, train_encoder
from dynalign.worlds import local_transition, sensor_table, training_observations


@pytest.mark.parametrize("experiment", ["sample_efficiency", "robustness", "pixels"])
def test_tiny_end_to_end_and_cache(tmp_path: Path, experiment: str):
    specs = grid(experiment, "smoke")[:1]
    specs[0]["steps"] = 3
    result = run_grid(specs, tmp_path, "cpu")
    assert len(result) == 1
    assert np.isfinite(result.iloc[0]["cka"])
    result_path = tmp_path / result.iloc[0]["run_id"] / "result.json"
    original_timestamp = result_path.stat().st_mtime_ns
    run_grid(specs, tmp_path, "cpu")
    assert result_path.stat().st_mtime_ns == original_timestamp
    saved = json.loads(result_path.read_text())
    assert saved["config"]["steps"] == 3
    assert (result_path.parent / "embeddings.npz").exists()


def test_checkpoint_sampler_resume_matches_uninterrupted(tmp_path: Path):
    torch.set_num_threads(1)
    config = TrainConfig(steps=4, checkpoint_every=2)
    data = training_observations(local_transition(), sensor_table(32, 0), 64, 0.05, 0)
    original = build_encoder((12,), config, 2)
    train_encoder(original, *data, config, 30, "cpu", tmp_path / "complete.pt")
    # A completed checkpoint reloaded with the identical config must preserve all weights.
    restored = build_encoder((12,), config, 99)
    train_encoder(restored, *data, config, 30, "cpu", tmp_path / "complete.pt")
    for a, b in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="configuration"):
        train_encoder(restored, *data, replace(config, steps=5), 30, "cpu", tmp_path / "complete.pt")


def test_grid_sizes_are_explicit():
    assert len(grid("sample_efficiency", "full")) == 90
    assert len(grid("robustness", "full")) == 60
    assert len(grid("pixels", "full")) == 60
    assert {row["steps"] for row in grid("sample_efficiency", "full")} == {800}


def test_interrupted_training_resume_is_exact(tmp_path: Path, monkeypatch):
    import dynalign.training as training

    torch.set_num_threads(1)
    config = TrainConfig(steps=6, checkpoint_every=2)
    data = training_observations(local_transition(), sensor_table(32, 0), 64, 0.05, 0)
    baseline = build_encoder((12,), config, 2)
    train_encoder(baseline, *data, config, 30, "cpu")
    interrupted = build_encoder((12,), config, 2)
    original_loss = training.vicreg_loss
    calls = 0

    def interrupt_after_checkpoint(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(training, "vicreg_loss", interrupt_after_checkpoint)
    checkpoint = tmp_path / "interrupted.pt"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_encoder(interrupted, *data, config, 30, "cpu", checkpoint)
    assert checkpoint.exists()
    monkeypatch.setattr(training, "vicreg_loss", original_loss)
    restored = build_encoder((12,), config, 999)
    train_encoder(restored, *data, config, 30, "cpu", checkpoint)
    for a, b in zip(baseline.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
