"""Verify whole-run resumption against an existing uninterrupted seed42 experiment."""

import json
from pathlib import Path

import torch

from ml_lab.engine import restore_checkpoint, train, write_json
from ml_lab.model import FashionMLP


def main():
    original = Path("runs/seed42")
    replay = Path("runs/resume-verification")
    # This is a dedicated replay: the full 10-epoch run is already complete.
    train("data/fashion-mnist", replay, epochs=5)
    train("data/fashion-mnist", replay, epochs=10, resume=True)
    comparisons = {}
    for filename in ["last.pt", "best.pt"]:
        a, b = FashionMLP(), FashionMLP()
        ca = restore_checkpoint(original / filename, a)
        cb = restore_checkpoint(replay / filename, b)
        equal = all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
        assert equal, f"Resumed weights differ: {filename}"
        ah = [{k: v for k, v in r.items() if k != "seconds"} for r in ca["history"]]
        bh = [{k: v for k, v in r.items() if k != "seconds"} for r in cb["history"]]
        assert ah == bh, f"Resumed metrics differ: {filename}"
        comparisons[filename] = {
            "all_model_tensors_bitwise_equal": equal,
            "epoch_metrics_equal_excluding_wall_time": ah == bh,
            "epoch": ca["epoch"],
        }
    result = {
        "protocol": "10 uninterrupted epochs vs 5 epochs + checkpoint restore + 5 epochs",
        "device": "cpu",
        "same_environment_required": True,
        "comparisons": comparisons,
    }
    write_json("results/seed42/resume_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
