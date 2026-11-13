"""Regression tests for host-side grading and tool boundaries."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from agent_v2_tools import Tools, bounded_process, compare


def test_file_tools_reject_escape_and_test_tampering(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "core.py").write_text("")
    (trial / "visible.json").write_text("[]")
    tools = Tools(trial)
    for path in ["../private.py", "visible.json", "_worker.py"]:
        result = tools.call(dict(tool="write_file", path=path, content="changed"))
        assert "error" in result
    assert tools.integrity()
    assert len(tools.violations) == 3
    assert (
        tools.call(dict(tool="write_file", path="core.py", content="x=1"))["written"] == "core.py"
    )


def test_symlink_to_protected_file_cannot_be_written(tmp_path):
    (tmp_path / "visible.json").write_text("[]")
    (tmp_path / "core.py").symlink_to(tmp_path / "visible.json")
    tools = Tools(tmp_path)
    result = tools.call(dict(tool="write_file", path="core.py", content="changed"))
    assert "error" in result
    assert json.loads((tmp_path / "visible.json").read_text()) == []


def test_host_grader_rejects_success_exit_mutation_and_bool_spoof():
    expected = [{"value": 1}]
    for actual in [
        {"status": "invalid_output"},
        {"status": "executed", "outputs": [{"value": 1, "mutated": True}]},
        {"status": "executed", "outputs": [{"value": True, "mutated": False}]},
        {"status": "executed", "outputs": [{"value": float("nan"), "mutated": False}]},
    ]:
        assert compare(actual, expected)["status"] != "pass"
    assert (
        compare({"status": "executed", "outputs": [{"value": 1, "mutated": False}]}, expected)[
            "status"
        ]
        == "pass"
    )


def test_output_flood_is_bounded_before_parsing():
    with pytest.raises(ValueError, match="1 MiB"):
        bounded_process([sys.executable, "-c", "print('x'*2000000)"], "[]", {}, None)
