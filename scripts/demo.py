"""Render actual predictions and the original seed42 Shirt error examples."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ml_lab.data import test_data
from ml_lab.engine import load_model, seed_everything, write_json

LABELS = [
    "T-shirt",
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


def main() -> None:
    """Use fixed official test order, without cherry-picking confidence or outcomes."""
    out = Path("results/demo")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    model, ck = load_model("runs/seed42")
    x, y = test_data("data/fashion-mnist", ck["data"]["mean"], ck["data"]["std"])
    with torch.inference_mode():
        logits = torch.cat([model(b) for b in x.split(512)])
    pred = logits.argmax(1)
    for name, indices in [
        ("predictions", torch.arange(12)),
        ("shirt_errors", torch.where((y == 6) & (pred == 0))[0][:12]),
    ]:
        fig, axes = plt.subplots(3, 4, figsize=(10, 8), layout="constrained")
        for ax, idx in zip(axes.flat, indices, strict=False):
            pixels = x[idx].reshape(28, 28) * ck["data"]["std"] + ck["data"]["mean"]
            confidence = logits[idx].softmax(0)[pred[idx]].item()
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"#{idx.item()} · actual {LABELS[y[idx]]}\npred {LABELS[pred[idx]]} ({confidence:.0%})",
                fontsize=9,
                color="#286344" if pred[idx] == y[idx] else "#a03227",
            )
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(
            "Original MLP seed42 · "
            + (
                "first 12 official test images"
                if name == "predictions"
                else "first 12 Shirt → T-shirt errors"
            )
        )
        fig.savefig(out / f"{name}.png", dpi=180)
        plt.close(fig)
    shirt = y == 6
    write_json(
        out / "summary.json",
        {
            "checkpoint_seed": 42,
            "shirt_recall": float((pred[shirt] == 6).float().mean()),
            "shirt_to_tshirt_count": int(((y == 6) & (pred == 0)).sum()),
            "image_selection": "first matching official test indices",
            "interpretation": "Similar low-resolution silhouettes and sleeve/neckline cues are a visual hypothesis, not a measured causal explanation.",
        },
    )
    print("Demo saved under results/demo; Shirt recall:", float((pred[shirt] == 6).float().mean()))


if __name__ == "__main__":
    main()
