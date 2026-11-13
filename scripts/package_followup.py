"""Package follow-up releases without rewriting any v1.0 release asset."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    """Archive committed source and separately verify new-capacity checkpoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--version", default="1.1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = root / "release-assets" / f"v{args.version}"
    assets.mkdir(parents=True, exist_ok=True)
    source = assets / f"ml-training-inference-lab-v{args.version}-source.zip"
    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            f"--prefix=ml-training-inference-lab-v{args.version}/",
            "--output",
            str(source),
            args.ref,
        ],
        cwd=root,
        check=True,
    )
    records = []
    for result in sorted((root / "results/capacity-sweep").glob("*.json")):
        row = json.loads(result.read_text())
        if "kind" not in row or "seed" not in row:
            continue
        path = root / f"runs/capacity-sweep/{row['kind']}-{row['seed']}/best.pt"
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if value != row["test"]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint mismatch: {path.name}")
        records.append((path, str(path.relative_to(root)), value))
    if len(records) != 12:
        raise ValueError("Require exactly twelve new-capacity checkpoints")
    with zipfile.ZipFile(
        assets / f"capacity-checkpoints-v{args.version}.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path, relative, _ in records:
            archive.write(path, relative)
        archive.writestr(
            "CHECKPOINT_SHA256.json", json.dumps({r: h for _, r, h in records}, indent=2) + "\n"
        )
    sums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.glob("*.zip"))
    }
    (assets / "SHA256SUMS.txt").write_text("".join(f"{h}  {name}\n" for name, h in sums.items()))
    (assets / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", args.ref], cwd=root, text=True
                ).strip(),
                "new_checkpoint_count": 12,
                "assets_sha256": sums,
                "v1_0_assets": "Unmodified; original checkpoints and runtime artifacts remain in release v1.0.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"directory": str(assets), "sha256": sums}, indent=2))


if __name__ == "__main__":
    main()
