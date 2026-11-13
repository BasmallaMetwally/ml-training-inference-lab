# V2 custom-tool repair experiment

Protocol and task files are fixed before formal collection. Twelve independent
multi-file tasks × two models × three repetitions = 72 planned trajectories.
See protocol.json for limits and source hashes. Preflight contains one excluded
engineering trial; it is never included in the formal score.

Only visible-test feedback is returned during repair. Final hidden grades run
in Docker in a private evaluator repository after every generation has ended.
Results will distinguish functional correctness, completion, protocol violations,
infrastructure failures, timeouts and crashes. Until grades exist, no v2 pass rate
or improvement over the saturated v1 pilot is claimed.

Local validation passed all 12 references and rejected all 12 starters. Filesystem,
network, mutation and host comparison controls have separate regression checks.
Twelve independent hand-calculated examples also validate the reference formulas.
The complete hidden suite is still private; hashes do not constitute full public
independent reproducibility.
