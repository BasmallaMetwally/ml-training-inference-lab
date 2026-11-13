# Corrected Linux memory follow-up

Three independent exec workers per condition; authoritative counter: `/proc/self/status` VmHWM.
Includes imports, runtime/model construction and 50 batch-128 forwards. Excludes compiler child processes; disk compiler caches may be warm. Process peaks are not activation-only memory.

| Runtime | Input checks | Peak RSS mean ± sample SD (MiB) | n |
| --- | --- | ---: | ---: |
| eager | True | 392.50 ± 0.04 | 3 |
| eager | False | 389.38 ± 0.14 | 3 |
| fused | True | 393.26 ± 0.24 | 3 |
| fused | False | 391.45 ± 0.29 | 3 |
| compiled | True | 475.20 ± 6.60 | 3 |
| compiled | False | 469.47 ± 0.07 | 3 |
| int8 | True | 395.77 ± 0.15 | 3 |
| int8 | False | 394.93 ± 0.38 | 3 |
| onnx | True | 446.29 ± 46.08 | 3 |
| onnx | False | 414.14 ± 0.11 | 3 |

The old constant 871 MiB table is inconclusive and is retained only for audit. These new observations supersede that comparison. Small differences are descriptive, not evidence of a reliable memory improvement.

[Raw stage counters](memory-corrected.json) · [Successful Linux workflow and parent-allocation regression](https://github.com/BasmallaMetwally/ml-training-inference-lab/actions/runs/43463432386)
