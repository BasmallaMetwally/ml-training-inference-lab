"""Preregistered paired 10-seed architecture study and error analysis."""

import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.benchmark import Timer

from ml_lab.data import test_data
from ml_lab.engine import final_evaluation, load_model, seed_everything, train, write_json
from ml_lab.model import fused_model

OUT = Path("results/study")
LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def summary(values):
    """Mean, sample standard deviation and Student-t interval across seeds."""
    a = np.asarray(values, dtype=float)
    spread = float(a.std(ddof=1))
    margin = float(stats.t.ppf(0.975, len(a) - 1) * spread / np.sqrt(len(a)))
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "std": spread,
        "ci95": [float(a.mean() - margin), float(a.mean() + margin)],
    }


def latency(model, batch):
    """Measure checked eval forward only, three steady-state repetitions."""
    sample = torch.zeros(batch, 784)
    medians = []
    with torch.inference_mode():
        for _ in range(20):
            model(sample)
        for _ in range(3):
            timer = Timer(
                "model(sample)", globals={"model": model, "sample": sample}, num_threads=1
            )
            medians.append(timer.blocked_autorange(min_run_time=0.05).median * 1000)
    return {"median_ms": float(np.median(medians)), "repeat_medians_ms": medians}


def error_analysis():
    """Analyze the predeclared representative seed 0, without tuning from errors."""
    model, ck = load_model("runs/study/mlp-0")
    x, y = test_data("data/fashion-mnist", ck["data"]["mean"], ck["data"]["std"])
    with torch.inference_mode():
        logits = torch.cat([model(xb) for xb in x.split(512)])
    pred = logits.argmax(1)
    confusion = torch.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    normalized = confusion.numpy() / confusion.sum(1).numpy()[:, None]
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set(
        xticks=range(10),
        yticks=range(10),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="MLP seed 0 · normalized test confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right")
    for i, j in itertools.product(range(10), repeat=2):
        ax.text(
            j,
            i,
            f"{normalized[i, j]:.0%}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if normalized[i, j] > 0.55 else "black",
        )
    fig.colorbar(im, ax=ax, label="Fraction of actual class")
    fig.savefig(OUT / "confusion.png", dpi=180)
    plt.close(fig)
    indices = torch.where((y == 6) & (pred == 0))[0][:16]
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), layout="constrained")
    records = []
    for ax, idx in zip(axes.flat, indices, strict=False):
        image = x[idx].reshape(28, 28) * ck["data"]["std"] + ck["data"]["mean"]
        confidence = logits[idx].softmax(0)[0].item()
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Test #{idx.item()}\nShirt → T-shirt ({confidence:.0%})", fontsize=9)
        records.append(
            {
                "test_index": idx.item(),
                "actual": "Shirt",
                "predicted": "T-shirt/top",
                "predicted_probability": confidence,
            }
        )
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("First 16 Shirt → T-shirt errors in official test order · MLP seed 0")
    fig.savefig(OUT / "shirt_errors.png", dpi=180)
    plt.close(fig)
    value = {
        "representative_seed": 0,
        "selection": "predeclared seed; first errors in official test order",
        "shirt_recall": normalized[6, 6],
        "shirt_to_tshirt_count": confusion[6, 0].item(),
        "confusion_matrix": confusion.tolist(),
        "examples": records,
        "interpretation": "Visual overlap in silhouettes, sleeves and low-resolution grayscale texture is a plausible explanation; no causal mechanism is established. Error inspection is post-evaluation and did not tune these models.",
    }
    write_json(OUT / "errors.json", value)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "seeds": list(range(10)),
        "architectures": ["mlp", "cnn"],
        "split_seed": 42,
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "selection": "minimum validation cross entropy",
        "test_policy": "same official test set for all predetermined runs; no test-driven tuning",
        "statistical_unit": "training seed, fixed dataset split, paired architectures",
        "primary_comparison": "CNN minus MLP test accuracy, paired exact sign-flip test and Student-t CI",
        "caveat": "Seed uncertainty is conditional on this split and is not a population-level data uncertainty interval.",
        "representative_seed": 0,
    }
    write_json(OUT / "plan.json", plan)
    records = []
    for seed, kind in itertools.product(range(10), ["mlp", "cnn"]):
        target = OUT / f"{kind}-{seed}.json"
        if target.exists():
            records.append(json.loads(target.read_text()))
            continue
        run = Path(f"runs/study/{kind}-{seed}")
        train("data/fashion-mnist", run, epochs=10, seed=seed, model_kind=kind, split_seed=42)
        metrics = final_evaluation("data/fashion-mnist", run)
        model, _ = load_model(run)
        seed_everything(seed)
        row = {
            "kind": kind,
            "seed": seed,
            "test": metrics,
            "parameters": sum(p.numel() for p in model.parameters()),
            "latency": {str(b): latency(model, b) for b in [1, 128]},
        }
        if kind == "mlp":
            fused = fused_model(model)
            row["fused_latency"] = {str(b): latency(fused, b) for b in [1, 128]}
        write_json(target, row)
        records.append(row)
        print(f"COMPLETED {kind} seed={seed} test={metrics['accuracy']:.4%}", flush=True)
    groups = {
        kind: sorted([r for r in records if r["kind"] == kind], key=lambda r: r["seed"])
        for kind in ["mlp", "cnn"]
    }
    a = np.array([r["test"]["accuracy"] for r in groups["mlp"]])
    b = np.array([r["test"]["accuracy"] for r in groups["cnn"]])
    diff = b - a
    null = np.array(
        [np.mean(diff * np.array(signs)) for signs in itertools.product([-1, 1], repeat=10)]
    )
    p = float(np.mean(np.abs(null) >= abs(diff.mean()) - 1e-15))
    result = {
        "plan": plan,
        "accuracy": {k: summary([r["test"]["accuracy"] for r in v]) for k, v in groups.items()},
        "paired_difference_cnn_minus_mlp": {
            **summary(diff),
            "exact_sign_flip_p_two_sided": p,
            "paired_t_p_two_sided": float(stats.ttest_rel(b, a).pvalue),
            "differences": diff.tolist(),
        },
        "latency_ms": {
            k: summary([r["latency"]["1"]["median_ms"] for r in v]) for k, v in groups.items()
        },
    }
    write_json(OUT / "summary.json", result)
    points = []
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for kind, color in [("mlp", "#2858b5"), ("cnn", "#b24b24")]:
        acc = result["accuracy"][kind]
        lat = result["latency_ms"][kind]
        params = groups[kind][0]["parameters"]
        ax.errorbar(
            lat["mean"],
            acc["mean"] * 100,
            xerr=lat["std"],
            yerr=acc["std"] * 100,
            color=color,
            capsize=5,
        )
        ax.scatter(
            lat["mean"],
            acc["mean"] * 100,
            s=params / 100,
            color=color,
            alpha=0.75,
            label=f"{kind.upper()} · {params:,} parameters",
        )
        ax.annotate(
            kind.upper(),
            (lat["mean"], acc["mean"] * 100),
            xytext=(10, 10),
            textcoords="offset points",
        )
        points.append(
            {"kind": kind, "latency_ms": lat["mean"], "accuracy": acc["mean"], "parameters": params}
        )
    for point in points:
        point["pareto_optimal_3objectives"] = not any(
            q["accuracy"] >= point["accuracy"]
            and q["latency_ms"] <= point["latency_ms"]
            and q["parameters"] <= point["parameters"]
            and (
                q["accuracy"] > point["accuracy"]
                or q["latency_ms"] < point["latency_ms"]
                or q["parameters"] < point["parameters"]
            )
            for q in points
        )
    write_json(
        OUT / "pareto.json",
        {"points": points, "objectives": "maximize accuracy, minimize latency and parameters"},
    )
    ax.set(
        xlabel="CPU batch-1 latency (ms), checked inference mode",
        ylabel="Official test accuracy (%)",
        title="Accuracy / latency / model size · 10 paired seeds\nMarker area ∝ parameters; error bars = sample standard deviation",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(OUT / "pareto.png", dpi=180)
    fig.savefig(OUT / "pareto.svg")
    plt.close(fig)
    error_analysis()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
