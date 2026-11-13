# V2 custom-tool repair experiment

Protocol and task files are fixed before formal collection. Twelve independent
multi-file tasks × two models × three repetitions = 72 planned trajectories.
See protocol.json for limits and source hashes. Preflight contains one excluded
engineering trial; it is never included in the formal score.

Only visible-test feedback is returned during repair. Final hidden grades run
in Docker in a private evaluator repository after every generation has ended.
Final results distinguish functional correctness, completion, protocol violations,
infrastructure failures, timeouts and crashes. See [the final report](REPORT.md):
accepted repairs are 26/36 versus 36/36; final functional correctness is 30/36
versus 36/36. V1 and V2 differ in tasks/tools and are not a causal comparison.

Local validation passed all 12 references and rejected all 12 starters. Filesystem,
network, mutation and host comparison controls have separate regression checks.
Twelve independent hand-calculated examples also validate the reference formulas.
The complete hidden suite is still private; hashes do not constitute full public
independent reproducibility.

To replay a final candidate on public fixtures without a model account:

```sh
docker build -f Dockerfile.agent-v2 -t ml-agent-v2:1 .
python scripts/replay_agent_v2.py TASK--MODEL--r0 --backend docker
```

This checks public visible fixtures only. It does not reproduce the withheld grade.
On macOS, `--backend macos` uses the same visible-test Seatbelt policy as collection.

Run `python scripts/validate_agent_v2_artifacts.py` to check the frozen task/harness hashes, all submitted candidates and unchanged hidden-case hashes. This integrity check runs in Linux/macOS CI.
