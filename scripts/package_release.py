"""Package canonical source separately from checkpoints and profiling artifacts."""

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def main() -> None:
    """Build a source-only Git archive plus separate, checksummed release assets."""
    root = Path(__file__).resolve().parents[1]
    # Machine hostname is not needed in a shareable correctness record.
    tests = ET.parse(root / "results/tests.xml")
    for suite in tests.iter("testsuite"):
        suite.attrib.pop("hostname", None)
    tests.write(root / "results/tests.xml", encoding="utf-8", xml_declaration=True)
    assets = root / "release-assets"
    assets.mkdir(exist_ok=True)
    checkpoints = [root / "runs/seed42/best.pt", root / "runs/seed42/last.pt"]
    checkpoints.extend(sorted((root / "runs/study").glob("*/best.pt")))
    checkpoint_manifest = {}
    with zipfile.ZipFile(assets / "checkpoints-v1.0.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in checkpoints:
            relative = str(path.relative_to(root))
            z.write(path, relative)
            checkpoint_manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        z.writestr("CHECKPOINT_SHA256.json", json.dumps(checkpoint_manifest, indent=2) + "\n")
    with zipfile.ZipFile(assets / "runtime-artifacts-v1.0.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted((root / "results/performance").glob("*-trace.json")):
            z.write(path, Path("macos") / path.name)
        for path in sorted((root / "results/performance").glob("*.onnx*")):
            z.write(path, Path("macos") / path.name)
        linux = root.parents[1] / "work/linux-runtime-final"
        for path in linux.rglob("*"):
            if path.is_file() and (path.name.endswith("-trace.json") or ".onnx" in path.name):
                z.write(path, Path("linux") / path.name)
    asset_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in assets.glob("*.zip")
    }
    (assets / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(asset_hashes.items()))
    )
    archive = root.parent / "ml-training-inference-lab.zip"
    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            "--prefix=ml-training-inference-lab/",
            "--output",
            str(archive),
            "HEAD",
        ],
        cwd=root,
        check=True,
    )
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(
            "/runs/" in name or "/data/" in name or name.endswith(".pt") for name in z.namelist()
        )
    print(f"{archive}: {archive.stat().st_size:,} bytes; no runs, dataset files or checkpoints")
    print("Separate release assets:", asset_hashes)


if __name__ == "__main__":
    main()
