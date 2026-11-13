"""Verify that published results still match frozen tasks and submitted code."""

import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results/agents-v2"
    protocol = json.loads((out / "protocol.json").read_text())
    for relative, value in protocol["suite_sha256"].items():
        assert sha(root / "tasks/agent_v2" / relative) == value, relative
    for name, value in protocol["harness_sha256"].items():
        assert sha(root / "scripts" / name) == value, name
    grades = json.loads((out / "grades.json").read_text())["grades"]
    expected = {(t, m, r) for t in protocol["tasks"] for m in protocol["models"] for r in range(3)}
    assert (
        len(grades) == 72 and {(g["task"], g["model"], g["repetition"]) for g in grades} == expected
    )
    for row in grades:
        folder = out / "trials" / row["trial"]
        metadata = json.loads((folder / "metadata.json").read_text())
        assert all(row[k] == v for k, v in metadata.items()), row["trial"]
        for name, value in row["candidate_sha256"].items():
            assert name in ["core.py", "pipeline.py"]
            assert sha(folder / "candidate" / name) == value, row["trial"]
        accepted = (
            row["functional"]["status"] == "pass"
            and row["status"] == "submitted"
            and row["protected_integrity"]
            and not row["violations"]
            and row["native_tool_events"] == 0
        )
        assert row["accepted"] == accepted, row["trial"]
    before = json.loads((out / "docker-prevalidation/v2_grader_validation.json").read_text())
    after = json.loads((out / "grader_validation.json").read_text())
    assert before["validation"] == after["validation"], (
        "Hidden cases/references changed after prevalidation"
    )
    assert sum(r["cases"] for r in after["validation"]) == 211
    print(
        "Verified: frozen tasks/harness, all 72 original candidates, acceptance policy and unchanged 211 hidden cases."
    )


if __name__ == "__main__":
    main()
