"""Record functional validation, not a coding-agent evaluation score."""

from pathlib import Path

from ml_lab.engine import write_json
from ml_lab.grader import grade


def main():
    cases = []
    for task in ["train_step", "stable_loss", "predict"]:
        for fixture, expected in [("reference.py", "pass"), ("starter.py", "incorrect")]:
            result = grade(task, Path("tasks") / fixture)
            assert result["status"] == expected, result
            cases.append({"fixture": fixture, **result})
    for task, fixture in [
        ("train_step", "stale_gradients.py"),
        ("stable_loss", "detached_loss.py"),
        ("predict", "no_eval.py"),
        ("predict", "no_inference.py"),
    ]:
        result = grade(task, Path("tasks/mutants") / fixture)
        assert result["status"] != "pass", result
        cases.append({"fixture": "mutants/" + fixture, **result})
    write_json(
        "results/task_validation.json",
        {
            "reference_passes": 3,
            "incorrect_fixtures_rejected": 7,
            "note": "Hand-authored fixture checks only. No external coding-agent trials.",
            "cases": cases,
        },
    )
    print("3 reference tasks passed; all 7 deliberately wrong fixtures rejected.")


if __name__ == "__main__":
    main()
