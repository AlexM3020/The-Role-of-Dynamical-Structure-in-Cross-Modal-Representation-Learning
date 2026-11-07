"""Independent unimodal training with resumable, fixed-update temporal VICReg."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 600
    batch_size: int = 128
    width: int = 64
    dim: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    invariance_weight: float = 25.0
    variance_weight: float = 25.0
    covariance_weight: float = 1.0
    checkpoint_every: int = 100


def build_encoder(input_shape: tuple[int, ...], config: TrainConfig, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if len(input_shape) == 1:
        return nn.Sequential(
            nn.Linear(input_shape[0], config.width), nn.SiLU(),
            nn.Linear(config.width, config.width), nn.SiLU(),
            nn.Linear(config.width, config.dim),
        )
    if len(input_shape) == 3:
        channels, height, width = input_shape
        return nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            nn.Flatten(), nn.Linear(16 * (height // 4) * (width // 4), config.width),
            nn.SiLU(), nn.Linear(config.width, config.dim),
        )
    raise ValueError(f"Unsupported input shape: {input_shape}")


def vicreg_loss(a: torch.Tensor, b: torch.Tensor, config: TrainConfig):
    if len(a) < 2:
        raise ValueError("Covariance requires a batch with at least two observations.")
    invariance = F.mse_loss(a, b)
    std_a = torch.sqrt(a.var(0, unbiased=True) + 1e-4)
    std_b = torch.sqrt(b.var(0, unbiased=True) + 1e-4)
    variance = (F.relu(1 - std_a).mean() + F.relu(1 - std_b).mean()) / 2
    ac, bc = a - a.mean(0), b - b.mean(0)
    cov_a, cov_b = ac.T @ ac / (len(a) - 1), bc.T @ bc / (len(b) - 1)
    off_diagonal = ~torch.eye(a.shape[1], dtype=torch.bool, device=a.device)
    covariance = (cov_a[off_diagonal].square().sum() + cov_b[off_diagonal].square().sum()) / a.shape[1]
    total = config.invariance_weight * invariance + config.variance_weight * variance + config.covariance_weight * covariance
    return total, {"invariance": invariance, "variance": variance, "covariance": covariance}


def encode(model: nn.Module, observations: np.ndarray, device: str, batch_size: int = 512):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            tensor = torch.as_tensor(observations[start:start + batch_size], device=device)
            outputs.append(model(tensor).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_encoder(
    model: nn.Module,
    first: np.ndarray,
    second: np.ndarray,
    config: TrainConfig,
    seed: int,
    device: str,
    checkpoint: Path | None = None,
):
    """Only observation arrays enter this function: no labels or other encoder."""
    if first.shape != second.shape or len(first) < 2:
        raise ValueError("Expected equally sized paired observation arrays with >=2 rows.")
    if config.steps < 1 or config.batch_size < 2:
        raise ValueError("Use positive steps and batch_size >= 2.")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Keep the complete small dataset on CPU; transfer only each minibatch.
    first_t, second_t = torch.from_numpy(first), torch.from_numpy(second)
    start_step, elapsed_before, history = 0, 0.0, []
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["config"] != asdict(config) or state["seed"] != seed:
            raise ValueError("Checkpoint configuration mismatch; use a new output directory.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["sampler_state"].cpu())
        start_step, elapsed_before, history = state["step"], state["elapsed_seconds"], state["history"]
    started = time.perf_counter()
    model.train()
    for step in range(start_step, config.steps):
        indices = torch.randint(len(first), (config.batch_size,), generator=generator)
        a = model(first_t[indices].to(device))
        b = model(second_t[indices].to(device))
        loss, components = vicreg_loss(a, b, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at update {step}.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        should_save = (step + 1) % config.checkpoint_every == 0 or step + 1 == config.steps
        if should_save:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu()),
                            **{key: float(value.detach().cpu()) for key, value in components.items()}})
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "config": asdict(config), "seed": seed, "step": step + 1,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "sampler_state": generator.get_state(), "history": history,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                }
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(state, temporary)
                os.replace(temporary, checkpoint)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return {"history": history, "training_seconds": elapsed_before + time.perf_counter() - started}
