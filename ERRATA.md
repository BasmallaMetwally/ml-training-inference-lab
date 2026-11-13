# v1.0 measurement errata

The archived v1.0 Linux peak-memory table is **inconclusive**. All ten entries
report 913,358,848 bytes although baseline/final RSS differ. The subprocess
launcher and `ru_maxrss` can retain pre-exec history, so those numbers do not
establish per-runtime memory usage or equality. The original JSON remains an
audit record; it is not valid comparative evidence. macOS does not show the
same constant-value symptom, which alone does not prove its measurements exact.

The corrected method reads Linux `/proc/self/status` VmHWM inside the worker,
records setup/inference stages, uses a standard-library-only launcher and repeats
each condition three times. A Linux regression test puts a 160 MiB allocation in
the parent and checks that the child's VmHWM is not contaminated by it.

Other interpretation limits: architecture seeds are indexed blocks, not evidence
of matched initialization or random draws across architectures; a Welch sensitivity
analysis accompanies the original index-paired comparison. The two-point Pareto
plot only describes the two measured architectures. Agent responses are six suite
generations, containing 72 correlated function-level attempts, with a ceiling effect.

The v1.0 Zenodo files and Git tag remain immutable. Corrections are recorded on
the main branch and linked from subsequent release metadata.
