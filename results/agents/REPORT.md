# Coding-agent repair pilot

Two models, three independent suite responses each, twelve tasks per response, identical prompt, medium reasoning effort, no tools or grader feedback. All 72 task attempts were evaluated inside Docker without network; host-side references were not mounted into candidate containers.

| Model | Passed attempts | Pass rate | Nominal attempt CI | All-repeat successful tasks | Task-level CI | Timeouts / crashes |
| --- | ---: | ---: | --- | ---: | --- | --- |
| gpt-6-luna | 36/36 | 100% | 90.4%–100.0% | 12/12 | 75.8%–100.0% | 0 / 0 |
| gpt-6.1-sol | 36/36 | 100% | 90.4%–100.0% | 12/12 | 75.8%–100.0% | 0 / 0 |

## Interpretation

No observed correctness failures, crashes, timeouts or tool-use violations. The suite has a ceiling effect for these two models; no ranking is established.

Attempt-level Wilson treats correlated outcomes as independent and is nominal only. Task-level all-repetitions Wilson uses 12 unique tasks, still assumes a Bernoulli sampling model for a purposively selected fixed suite. Neither interval supports population-wide agent ranking.

The attempt intervals are displayed for transparency, not as independent-sample evidence. A task appears three times in the same model condition and functions share suite responses. These are code-generation repair trials, not multi-step tool-using agent trajectories.

## Failure and reward-hacking analysis

No observed test access, file/network access or grader modification in the recorded zero-tool trials and inspected solutions. This is not an adversarial security certification.

There are no observed model failure categories to compare. Missing a failure does not establish that the failure cannot occur. Deliberately broken starters are separately rejected by the withheld evaluator, so successful generation is not being confused with successful grading.

## Evidence

- `grades.json`: task statuses and candidate source hashes, without hidden inputs.
- `grader_validation.json`: twelve independent reference functions pass and twelve broken starters are rejected.
- Per-trial folders: complete prompt response, executable solution, transcript and generation metadata.
- `protocol.json`: models, repetition count and budgets.

Docker: no network, read-only root and submission, unprivileged user, no capabilities, no new privileges, limited PIDs/CPU/RAM and a timeout. The grader and reference calculations run on the host in a separate private repository.
