import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import digest, test_data, training_data
from .model import make_model


def environment() -> dict:
    """Record numerical runtime, platform and determinism settings."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def seed_everything(seed: int, threads: int = 1) -> None:
    """Set all experiment RNGs and the CPU numerical policy."""
    if threads < 1:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def write_json(path: str | Path, value: object) -> None:
    """Atomically write JSON, rejecting non-finite numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def train_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor
) -> tuple[float, float]:
    """Perform one checked, clipped-gradient classification update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite model parameters")
    return loss.item(), float(norm)


@torch.inference_mode()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> dict:
    """Return loss, accuracy and class diagnostics without changing BN statistics."""
    if batch_size < 1 or len(x) == 0 or len(x) != len(y):
        raise ValueError("Invalid evaluation data or batch size")
    model.eval()
    loss, correct = 0.0, 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for start in range(0, len(x), batch_size):
        xb, yb = x[start : start + batch_size], y[start : start + batch_size]
        logits = model(xb)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite predictions")
        loss += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        confusion += torch.bincount(yb * 10 + pred, minlength=100).reshape(10, 10)
    recall = confusion.diag().float() / confusion.sum(1).clamp_min(1)
    return {
        "loss": loss / len(x),
        "accuracy": correct / len(x),
        "count": len(x),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    epoch: int,
    best_loss: float,
    config: dict,
    metadata: dict,
    history: list,
) -> None:
    """Persist all state used by the CPU training trajectory."""
    # CPU-only experiment: preserve every RNG used by training, including dropout and shuffle.
    value = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": torch.get_rng_state(),
        "shuffle_rng": generator.get_state(),
        "epoch": epoch,
        "best_loss": best_loss,
        "config": config,
        "data": metadata,
        "history": history,
        "schema_version": 1,
    }
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def restore_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Restore model state and optionally optimizer and RNG state."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["rng"])
        if generator is None:
            raise ValueError("Resuming requires the shuffle generator")
        generator.set_state(ck["shuffle_rng"])
    return ck


def train(
    data_dir: str | Path,
    run_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 0.001,
    seed: int = 42,
    threads: int = 1,
    resume: bool = False,
    model_kind: str = "mlp",
    split_seed: int | None = None,
) -> list:
    """Train to a total epoch count, selecting checkpoints by validation loss."""
    if epochs < 1 or batch_size < 2 or lr <= 0 or not math.isfinite(lr):
        raise ValueError("Require positive epochs/lr and batch_size >= 2")
    run_dir = Path(run_dir)
    if (run_dir / "last.pt").exists() and not resume:
        raise ValueError("Run already exists; choose a new directory or use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, threads)
    x, y, vx, vy, meta = training_data(data_dir, seed if split_seed is None else split_seed)
    config = {
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "threads": threads,
        "architecture": "784-128-BN-ReLU-Dropout-64-BN-ReLU-Dropout-10",
        "optimizer": "Adam",
        "selection": "minimum validation cross entropy",
    }
    if model_kind != "mlp":
        config["model_kind"] = model_kind
        config["architecture"] = {
            "cnn": "Conv8-ReLU-Pool-Conv16-ReLU-Pool-Linear10",
            "mlp_small": "784-64-BN-ReLU-Dropout-32-BN-ReLU-Dropout-10",
            "mlp_large": "784-256-BN-ReLU-Dropout-128-BN-ReLU-Dropout-10",
            "cnn_small": "Conv4-ReLU-Pool-Conv8-ReLU-Pool-Linear10",
            "cnn_large": "Conv12-ReLU-Pool-Conv24-ReLU-Pool-Linear10",
        }[model_kind]
    if split_seed is not None:
        config["split_seed"] = split_seed
    model = make_model(model_kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    start, best, history = 0, float("inf"), []
    if resume:
        ck = restore_checkpoint(run_dir / "last.pt", model, optimizer, generator)
        if ck["config"] != config or ck["data"] != meta:
            raise ValueError("Resume configuration/data do not match the checkpoint")
        if not (run_dir / "best.pt").exists():
            raise ValueError("Resume requires the previous best checkpoint")
        start, best, history = ck["epoch"], ck["best_loss"], ck["history"]
        if epochs <= start:
            raise ValueError("epochs must exceed the last completed epoch")
    write_json(run_dir / "environment.json", environment())
    write_json(run_dir / "config.json", {**config, "target_epochs": epochs})
    write_json(run_dir / "data.json", meta)
    for epoch in range(start + 1, epochs + 1):
        before = time.perf_counter()
        order = torch.randperm(len(x), generator=generator)
        train_loss, max_grad = 0.0, 0.0
        batches = list(order.split(batch_size))
        if len(batches[-1]) == 1:  # BatchNorm cannot train on a singleton.
            tail = batches.pop()
            batches[-1] = torch.cat((batches[-1], tail))
        for idx in batches:
            loss, norm = train_step(model, optimizer, x[idx], y[idx])
            train_loss += loss * len(idx)
            max_grad = max(max_grad, norm)
        validation = evaluate(model, vx, vy)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(x),
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "max_preclip_gradient_norm": max_grad,
                "seconds": time.perf_counter() - before,
            }
        )
        improved = validation["loss"] < best
        best = min(best, validation["loss"])
        save_checkpoint(
            run_dir / "last.pt", model, optimizer, generator, epoch, best, config, meta, history
        )
        if improved:
            save_checkpoint(
                run_dir / "best.pt", model, optimizer, generator, epoch, best, config, meta, history
            )
        write_json(run_dir / "history.json", history)
        print(
            f"epoch {epoch:02d}: train={train_loss / len(x):.4f} val={validation['loss']:.4f} accuracy={validation['accuracy']:.4f}",
            flush=True,
        )
    return history


def load_model(run_dir: str | Path) -> tuple[nn.Module, dict]:
    """Construct and load the selected architecture in eval mode."""
    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True)
    model = make_model(ck["config"].get("model_kind", "mlp"))
    ck = restore_checkpoint(Path(run_dir) / "best.pt", model)
    model.eval()
    return model, ck


def final_evaluation(data_dir: str | Path, run_dir: str | Path) -> dict:
    """Evaluate a validation-selected checkpoint on the official test split."""
    run_dir = Path(run_dir)
    model, ck = load_model(run_dir)
    seed_everything(ck["config"]["seed"], ck["config"]["threads"])
    x, y = test_data(data_dir, ck["data"]["mean"], ck["data"]["std"])
    metrics = evaluate(model, x, y)
    metrics.update(
        {
            "selected_epoch": ck["epoch"],
            "checkpoint_sha256": digest(run_dir / "best.pt"),
            "selection": "validation loss only; official test split used for final evaluation",
            "majority_class_baseline_accuracy": y.bincount().max().item() / len(y),
        }
    )
    write_json(run_dir / "test_metrics.json", metrics)
    return metrics
