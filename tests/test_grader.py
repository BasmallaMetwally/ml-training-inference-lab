from pathlib import Path

import pytest

from ml_lab.grader import grade

TASKS = Path(__file__).resolve().parents[1] / "tasks"


@pytest.mark.parametrize("task", ["train_step", "stable_loss", "predict"])
def test_reference_passes(task):
    assert grade(task, TASKS / "reference.py")["status"] == "pass"


@pytest.mark.parametrize("task", ["train_step", "stable_loss", "predict"])
def test_starter_is_rejected(task):
    assert grade(task, TASKS / "starter.py")["status"] == "incorrect"


@pytest.mark.parametrize(
    "task,filename",
    [
        ("train_step", "stale_gradients.py"),
        ("stable_loss", "detached_loss.py"),
        ("predict", "no_eval.py"),
        ("predict", "no_inference.py"),
    ],
)
def test_plausible_partial_fixes_are_rejected(task, filename):
    assert grade(task, TASKS / "mutants" / filename)["status"] != "pass"


def test_timeout_is_classified(tmp_path):
    path = tmp_path / "slow.py"
    path.write_text("import time\ntime.sleep(20)\n")
    assert grade("predict", path, timeout=0.5)["status"] == "timeout"


def test_crash_is_classified(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text('raise RuntimeError("fixture crash")\n')
    assert grade("predict", path)["status"] == "crash"
