"""Exploratory capacity tradeoffs: six architectures on three common run blocks."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.benchmark import Timer

from ml_lab.data import digest
from ml_lab.engine import final_evaluation, load_model, seed_everything, train, write_json
from ml_lab.statistics import summarize


def main() -> None:
    """Keep the original 10-seed study intact; label this post-hoc sweep separately."""
    out = Path("results/capacity-sweep")
    out.mkdir(parents=True, exist_ok=True)
    kinds = ["mlp_small", "mlp", "mlp_large", "cnn_small", "cnn", "cnn_large"]
    write_json(
        out / "plan.json",
        {
            "architectures": kinds,
            "seeds": [0, 1, 2],
            "split_seed": 42,
            "epochs": 10,
            "batch_size": 128,
            "lr": 0.001,
            "cpu_threads": 1,
            "status": "Exploratory follow-up after observing v1.0 test results; not a new untouched-test confirmation.",
            "plot": "All six points use the same three seeds; original 10-seed inference remains separate.",
        },
    )
    records = []
    for kind in kinds:
        for seed in range(3):
            baseline = kind in ["mlp", "cnn"]
            run = Path(f"runs/{'study' if baseline else 'capacity-sweep'}/{kind}-{seed}")
            existing = (
                Path(f"results/study/{kind}-{seed}.json")
                if baseline
                else out / f"{kind}-{seed}.json"
            )
            if existing.exists() and (run / "best.pt").exists():
                row = json.loads(existing.read_text())
                if row["test"]["checkpoint_sha256"] != digest(run / "best.pt"):
                    raise ValueError("Checkpoint mismatch")
                records.append(row)
                continue
            train("data/fashion-mnist", run, epochs=10, seed=seed, model_kind=kind, split_seed=42)
            metrics = final_evaluation("data/fashion-mnist", run)
            model, _ = load_model(run)
            seed_everything(seed)
            times = []
            sample = torch.zeros(1, 784)
            with torch.inference_mode():
                for _ in range(20):
                    model(sample)
                for _ in range(5):
                    m = Timer(
                        "m(x)", globals={"m": model, "x": sample}, num_threads=1
                    ).blocked_autorange(min_run_time=0.1)
                    times.append(m.median * 1000)
            row = {
                "kind": kind,
                "seed": seed,
                "test": metrics,
                "parameters": sum(p.numel() for p in model.parameters()),
                "latency": {
                    "1": {"median_ms": float(np.median(times)), "repeat_medians_ms": times}
                },
            }
            write_json(existing, row)
            records.append(row)
            print("COMPLETED", kind, seed, metrics["accuracy"], flush=True)
    # Measure every checkpoint after training under one interleaved protocol.
    import random

    models = {}
    timings = {}
    sample = torch.zeros(1, 784)
    for row in records:
        key = (row["kind"], row["seed"])
        group = "study" if key[0] in ["mlp", "cnn"] else "capacity-sweep"
        models[key], _ = load_model(Path(f"runs/{group}/{key[0]}-{key[1]}"))
        timings[key] = []
    with torch.inference_mode():
        for model in models.values():
            for _ in range(20):
                model(sample)
        rng = random.Random(42)
        for _ in range(5):
            order = list(models)
            rng.shuffle(order)
            for key in order:
                measurement = Timer(
                    "m(x)", globals={"m": models[key], "x": sample}, num_threads=1
                ).blocked_autorange(min_run_time=0.1)
                timings[key].append(measurement.median * 1000)
    matched = []
    for row in records:
        values = timings[(row["kind"], row["seed"])]
        row["latency"] = {"1": {"median_ms": float(np.median(values))}}
        matched.append(
            {
                "kind": row["kind"],
                "seed": row["seed"],
                "repeat_medians_ms": values,
                "checkpoint_sha256": row["test"]["checkpoint_sha256"],
            }
        )
    write_json(
        out / "latency-matched.json",
        {
            "protocol": "All 18 checkpoints after training; checked CPU batch 1; zero input; 20 warmups; five interleaved repeats in fixed shuffled order; blocked_autorange 0.1s; one thread.",
            "measurements": matched,
        },
    )
    points = []
    for kind in kinds:
        rows = [r for r in records if r["kind"] == kind]
        points.append(
            {
                "kind": kind,
                "parameters": rows[0]["parameters"],
                "accuracy": summarize([r["test"]["accuracy"] for r in rows]),
                "latency_ms": summarize([r["latency"]["1"]["median_ms"] for r in rows]),
            }
        )
    for p in points:
        p["pareto_3objectives"] = not any(
            q["accuracy"]["mean"] >= p["accuracy"]["mean"]
            and q["latency_ms"]["mean"] <= p["latency_ms"]["mean"]
            and q["parameters"] <= p["parameters"]
            and (
                q["accuracy"]["mean"] > p["accuracy"]["mean"]
                or q["latency_ms"]["mean"] < p["latency_ms"]["mean"]
                or q["parameters"] < p["parameters"]
            )
            for q in points
        )
    write_json(
        out / "summary.json",
        {
            "points": points,
            "scope": "Descriptive three-objective nondominance among these six measured architectures only; post-hoc exploratory test-set reuse.",
        },
    )
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    for p in points:
        color = "#215fa4" if p["kind"].startswith("mlp") else "#b64d2d"
        x, y = p["latency_ms"]["mean"], p["accuracy"]["mean"] * 100
        ax.errorbar(
            x,
            y,
            xerr=p["latency_ms"]["std"],
            yerr=p["accuracy"]["std"] * 100,
            color=color,
            capsize=4,
        )
        ax.scatter(
            x,
            y,
            s=p["parameters"] / 140,
            color=color,
            alpha=0.65,
            edgecolors="black" if p["pareto_3objectives"] else "none",
            linewidths=2,
        )
        ax.annotate(
            f"{p['kind']}\n{p['parameters']:,} params",
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(
        xlabel="CPU batch-1 checked latency (ms)",
        ylabel="Official test accuracy (%)",
        title="Exploratory capacity sweep · six architectures · three common seeds\nMarker area ∝ parameters; bars = SD; black outline = nondominated among these six",
    )
    ax.margins(0.16)
    ax.grid(alpha=0.2)
    fig.savefig(out / "pareto.png", dpi=180)
    fig.savefig(out / "pareto.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
