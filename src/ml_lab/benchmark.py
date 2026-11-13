"""CPU steady-state forward benchmarks, excluding data loading and model setup."""

import contextlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.benchmark import Timer

from .data import digest, training_data
from .engine import environment, evaluate, load_model, seed_everything, write_json
from .model import fused_model


def benchmark(
    data_dir: str | Path,
    run_dir: str | Path,
    batch_sizes: tuple[int, ...] = (1, 32, 128, 512),
    repeats: int = 5,
    min_run_time: float = 0.1,
) -> dict:
    """Check fusion parity and collect randomized CPU forward timing blocks."""
    if repeats < 3 or min_run_time <= 0 or any(b < 1 for b in batch_sizes):
        raise ValueError("Use positive batch sizes/runtime and at least three repeats")
    model, ck = load_model(run_dir)
    seed_everything(ck["config"]["seed"], ck["config"]["threads"])
    _, _, x, y, meta = training_data(data_dir, ck["config"].get("split_seed", ck["config"]["seed"]))
    if meta != ck["data"]:
        raise ValueError("Benchmark data do not match the trained checkpoint")
    fused = fused_model(model)
    # Measure equivalence on validation data; test data do not influence optimization.
    max_error, disagreements = 0.0, 0
    with torch.inference_mode():
        for xb in x.split(512):
            expected, actual = model(xb), fused(xb)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            max_error = max(max_error, (actual - expected).abs().max().item())
            disagreements += (actual.argmax(1) != expected.argmax(1)).sum().item()
    if disagreements:
        raise AssertionError("Fusion changed a validation class prediction")
    variants = {
        "eval_autograd": (model, False),
        "eval_inference_mode": (model, True),
        "fused_inference_mode": (fused, True),
    }
    measurements = {(name, b): [] for name in variants for b in batch_sizes}
    order_rng = random.Random(123)
    for repeat in range(repeats):
        jobs = list(measurements)
        order_rng.shuffle(jobs)
        for name, batch in jobs:
            if batch > len(x):
                raise ValueError("Batch size exceeds validation examples")
            m, inference = variants[name]
            sample = x[:batch].contiguous()
            context = torch.inference_mode if inference else contextlib.nullcontext
            # Context entry is outside each timed forward, as in a serving loop.
            with context():
                for _ in range(20):
                    m(sample)
                timing = Timer(
                    stmt="model(sample)",
                    globals={"model": m, "sample": sample},
                    num_threads=ck["config"]["threads"],
                ).blocked_autorange(min_run_time=min_run_time)
            measurements[name, batch].append(
                {
                    "repeat": repeat,
                    "median_seconds": timing.median,
                    "iqr_seconds": timing.iqr,
                    "number_per_run": timing.number_per_run,
                    "raw_block_seconds": timing.raw_times,
                }
            )
    rows = []
    for (name, batch), trials in measurements.items():
        seconds = np.array([t["median_seconds"] for t in trials])
        median = float(np.median(seconds))
        rows.append(
            {
                "variant": name,
                "batch_size": batch,
                "median_batch_ms": median * 1000,
                "min_repeat_ms": float(seconds.min() * 1000),
                "max_repeat_ms": float(seconds.max() * 1000),
                "images_per_second": batch / median,
                "trials": trials,
            }
        )
    result = {
        "environment": environment(),
        "checkpoint_sha256": digest(Path(run_dir) / "best.pt"),
        "protocol": {
            "warmup_forwards": 20,
            "repeats": repeats,
            "min_block_runtime_seconds": min_run_time,
            "order": "randomized interleaving, seed 123",
            "scope": "CPU forward only, data already in memory",
            "includes": "input validation and output allocation",
            "excludes": "data loading, preprocessing, training, checkpoint loading, context entry",
            "timing": "torch.utils.benchmark.Timer.blocked_autorange; median of repeat medians",
        },
        "equivalence": {
            "validation_count": len(x),
            "max_abs_logit_error": max_error,
            "prediction_disagreements": disagreements,
            "atol": 1e-4,
            "rtol": 1e-4,
            "original_accuracy": evaluate(model, x, y)["accuracy"],
            "fused_accuracy": evaluate(fused, x, y)["accuracy"],
        },
        "parameter_storage_bytes": {
            "original": sum(p.numel() * p.element_size() for p in model.parameters()),
            "fused": sum(p.numel() * p.element_size() for p in fused.parameters()),
        },
        "memory_note": "Parameter tensor storage only; not peak process memory or activation memory.",
        "measurements": rows,
    }
    write_json(Path(run_dir) / "benchmark.json", result)
    return result
