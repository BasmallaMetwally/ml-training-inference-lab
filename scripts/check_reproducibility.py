"""Record same-platform exact resumption and cross-platform tensor fingerprints."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from ml_lab.data import download
from ml_lab.engine import environment, seed_everything, train, write_json


def fingerprints(checkpoint: Path) -> dict[str, str]:
    """Hash model tensors, excluding serialization headers and filesystem metadata."""
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {
        name: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        for name, tensor in value["model"].items()
    }


def main() -> None:
    """Run two full-data CPU trajectories and save both positive and negative findings."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/reproducibility/local.json")
    p.add_argument("--reference", default="results/reproducibility/macos.json")
    a = p.parse_args()
    download("data/fashion-mnist")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        train("data/fashion-mnist", root / "full", epochs=2)
        train("data/fashion-mnist", root / "resumed", epochs=1)
        train("data/fashion-mnist", root / "resumed", epochs=2, resume=True)
        first = fingerprints(root / "full/last.pt")
        second = fingerprints(root / "resumed/last.pt")
    seed_everything(42)
    result = {
        "protocol": "2 full-data epochs vs 1 + restore + 1; seed42, 1 CPU thread",
        "environment": environment(),
        "same_platform_bitwise_equal": first == second,
        "model_tensor_sha256": first,
    }
    reference = Path(a.reference)
    if reference.exists():
        expected = json.loads(reference.read_text())["model_tensor_sha256"]
        result["matches_recorded_macos_cpu"] = first == expected
        result["cross_platform_mismatching_tensors"] = [
            k for k, v in first.items() if expected.get(k) != v
        ]
    write_json(a.out, result)
    print(json.dumps(result, indent=2))
    if first != second:
        raise SystemExit("Same-platform exact resumption failed; result was recorded.")


if __name__ == "__main__":
    main()
