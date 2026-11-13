"""Summarize actual withheld grades while exposing the small-suite ceiling."""

import ast
import hashlib
import json
from pathlib import Path

from scipy.stats import norm

from ml_lab.engine import write_json


def wilson(successes: int, total: int) -> list[float]:
    """Nominal Wilson interval; the sampling assumptions must be reported."""
    z = float(norm.ppf(0.975))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def main() -> None:
    """Aggregate fixed-suite outcomes and audit immutable solution/transcript records."""
    root = Path("results/agents")
    grades = json.loads((root / "grades.json").read_text())
    models = {}
    audit = []
    for trial in grades["trials"]:
        folder = root / f"{trial['model']}-r{trial['repetition']}"
        metadata = json.loads((folder / "metadata.json").read_text())
        source = (folder / "solution.py").read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == trial["source_sha256"]
        assert metadata["tool_events"] == 0
        imports = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        audit.append(
            {
                "model": trial["model"],
                "repetition": trial["repetition"],
                "source_sha256": trial["source_sha256"],
                "imports": imports,
                "recorded_tool_events": 0,
            }
        )
    for model in sorted({t["model"] for t in grades["trials"]}):
        trials = [t for t in grades["trials"] if t["model"] == model]
        outcomes = [r for t in trials for r in t["outcomes"]]
        passed = sum(r["status"] == "pass" for r in outcomes)
        tasks = sorted({r["task"] for r in outcomes})
        repeat_successes = {
            task: sum(r["status"] == "pass" for r in outcomes if r["task"] == task)
            for task in tasks
        }
        all_repetitions_passed = sum(n == len(trials) for n in repeat_successes.values())
        models[model] = {
            "task_attempts": len(outcomes),
            "passes": passed,
            "pass_rate": passed / len(outcomes),
            "nominal_attempt_wilson_ci95": wilson(passed, len(outcomes)),
            "tasks_passing_all_repetitions": all_repetitions_passed,
            "unique_tasks": len(tasks),
            "task_all_repetitions_wilson_ci95": wilson(all_repetitions_passed, len(tasks)),
            "timeouts": sum(r["status"] == "timeout" for r in outcomes),
            "crashes": sum(r["status"] == "crash" for r in outcomes),
            "per_task_passes": repeat_successes,
        }
    result = {
        "models": models,
        "source_audit": audit,
        "interval_caveat": "Attempt-level Wilson treats correlated outcomes as independent and is nominal only. Task-level all-repetitions Wilson uses 12 unique tasks, still assumes a Bernoulli sampling model for a purposively selected fixed suite. Neither interval supports population-wide agent ranking.",
        "failure_analysis": "No observed correctness failures, crashes, timeouts or tool-use violations. The suite has a ceiling effect for these two models; no ranking is established.",
        "reward_hacking": "No observed test access, file/network access or grader modification in the recorded zero-tool trials and inspected solutions. This is not an adversarial security certification.",
        "grader_validation_run": "https://github.com/BasmallaMetwally/ml-lab-private-evaluator/actions/runs/43451808913",
        "public_results_exclude": "Withheld case inputs and reference calculations",
    }
    write_json(root / "summary.json", result)
    lines = [
        "# Coding-agent repair pilot",
        "",
        "Two models, three independent suite responses each, twelve tasks per response, identical prompt, medium reasoning effort, no tools or grader feedback. All 72 task attempts were evaluated inside Docker without network; host-side references were not mounted into candidate containers.",
        "",
        "| Model | Passed attempts | Pass rate | Nominal attempt CI | All-repeat successful tasks | Task-level CI | Timeouts / crashes |",
        "| --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for model, r in models.items():
        a, b = r["nominal_attempt_wilson_ci95"]
        c, d = r["task_all_repetitions_wilson_ci95"]
        lines.append(
            f"| {model} | {r['passes']}/{r['task_attempts']} | {r['pass_rate']:.0%} | {a:.1%}–{b:.1%} | {r['tasks_passing_all_repetitions']}/{r['unique_tasks']} | {c:.1%}–{d:.1%} | {r['timeouts']} / {r['crashes']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        result["failure_analysis"],
        "",
        result["interval_caveat"],
        "",
        "The attempt intervals are displayed for transparency, not as independent-sample evidence. A task appears three times in the same model condition and functions share suite responses. These are code-generation repair trials, not multi-step tool-using agent trajectories.",
        "",
        "## Failure and reward-hacking analysis",
        "",
        result["reward_hacking"],
        "",
        "There are no observed model failure categories to compare. Missing a failure does not establish that the failure cannot occur. Deliberately broken starters are separately rejected by the withheld evaluator, so successful generation is not being confused with successful grading.",
        "",
        "## Evidence",
        "",
        "- `grades.json`: task statuses and candidate source hashes, without hidden inputs.",
        "- `grader_validation.json`: twelve independent reference functions pass and twelve broken starters are rejected.",
        "- Per-trial folders: complete prompt response, executable solution, transcript and generation metadata.",
        "- `protocol.json`: models, repetition count and budgets.",
        "",
        "Docker: no network, read-only root and submission, unprivileged user, no capabilities, no new privileges, limited PIDs/CPU/RAM and a timeout. The grader and reference calculations run on the host in a separate private repository.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
