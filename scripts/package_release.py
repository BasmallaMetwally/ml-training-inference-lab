"""Snapshot evidence and create a portable source/checkpoint archive."""

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ml_lab.report import report


def main():
    root = Path(__file__).resolve().parents[1]
    # Machine hostname is not needed in a shareable correctness record.
    tests = ET.parse(root / "results/tests.xml")
    for suite in tests.iter("testsuite"):
        suite.attrib.pop("hostname", None)
    tests.write(root / "results/tests.xml", encoding="utf-8", xml_declaration=True)
    source, target = root / "runs/seed42", root / "results/seed42"
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "config.json",
        "data.json",
        "environment.json",
        "history.json",
        "test_metrics.json",
        "benchmark.json",
    ]:
        shutil.copy2(source / name, target / name)
    shutil.copy2(root / "data/fashion-mnist/manifest.json", target / "dataset_manifest.json")
    report(source, target)
    manifest = {}
    for folder in ["src", "tests", "scripts", "tasks", ".github"]:
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                manifest[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    for name in ["pyproject.toml", "requirements-lock.txt"]:
        manifest[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (target / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = root.parent / "ml-training-inference-lab.zip"
    count = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(
                p in {"__pycache__", ".pytest_cache", ".git", ".venv"} or p.endswith(".egg-info")
                for p in rel.parts
            ):
                continue
            if rel.parts[0] == "data":
                continue
            if rel.parts[0] == "runs" and rel.parts[1] != "seed42":
                continue
            z.write(path, Path(root.name) / rel)
            count += 1
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(f"{archive}: {count} files, {archive.stat().st_size:,} bytes; integrity verified")


if __name__ == "__main__":
    main()
