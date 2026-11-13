"""Recheck a published final candidate on the original public visible fixtures."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from agent_v2_tools import compare, digest, execute


def main():
    """Replay requires no model account and makes no hidden-grader claims."""
    parser = argparse.ArgumentParser()
    parser.add_argument("trial")
    parser.add_argument("--backend", choices=["macos", "docker"], default="docker")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trials = (root / "results/agents-v2/trials").resolve()
    trial = (trials / args.trial).resolve()
    if trial.parent != trials:
        raise ValueError("Trial must be inside the published trials directory")
    metadata = json.loads((trial / "metadata.json").read_text())
    for name, value in metadata["candidate_sha256"].items():
        if name not in ["core.py", "pipeline.py"] or digest(trial / "candidate" / name) != value:
            raise ValueError("Candidate hash mismatch")
    fixtures = json.loads((root / "tasks/agent_v2" / metadata["task"] / "visible.json").read_text())
    with tempfile.TemporaryDirectory(prefix="ml-agent-v2-visible-") as directory:
        workspace = Path(directory)
        for name in ["core.py", "pipeline.py"]:
            shutil.copyfile(trial / "candidate" / name, workspace / name)
        shutil.copyfile(root / "scripts/agent_v2_worker.py", workspace / "_worker.py")
        # The container's unprivileged user must be able to traverse this bind mount.
        workspace.chmod(0o755)
        result = compare(
            execute(workspace, [r["input"] for r in fixtures], args.backend),
            [r["expected"] for r in fixtures],
        )
    print(
        json.dumps(
            {"trial": args.trial, "scope": "public visible fixtures only", "result": result},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
