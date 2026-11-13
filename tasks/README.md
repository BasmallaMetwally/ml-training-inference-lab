# ML debugging tasks

Three small repair tasks connect the training project to coding-agent evaluation.

| Function | Contract |
| --- | --- |
| `train_step(model, optimizer, x, y)` | One classification update; clear old gradients, compute mean cross entropy, backpropagate and update parameters; return a finite Python float. |
| `stable_loss(logits, targets)` | Return differentiable, numerically stable mean cross entropy for finite float32 logits, including large positive/negative values. |
| `predict(model, x)` | Return eval-mode logits with autograd disabled; must not update BatchNorm running statistics. Leaving the model in eval mode is allowed. |

## Try a repair

Copy `starter.py` to a separate workspace, repair one function and run:

```bash
ml-lab grade --task train_step --candidate /absolute/path/to/solution.py --out results/agent-trial.json
```

Other task names are `stable_loss` and `predict`. Exit code 0 means pass; 1 means a rejected solution. Results distinguish incorrect output, crash and timeout. Each grader checks three seeded cases. The starter fails all three tasks; reference solutions pass. Additional mutants probe missing gradient clearing, detached losses and incomplete inference fixes.

## Coding-agent experiment protocol

1. Give the agent only the function contract and a copy of `starter.py`; keep `reference.py`, mutants and grader implementation outside its workspace.
2. Use a fresh workspace for every trial. Record agent/model version, prompt, time budget, tool budget, edits and command transcript.
3. Run the candidate with the external grader; save the result alongside its source and transcript. Declare whether feedback/retries were permitted.
4. Compare actual trials only; count timeouts and crashes in the denominator. Do not infer an agent pass rate from reference or mutant tests.

The grader code and seeds are public in this repository. These are development tasks, not a secure hidden benchmark. The subprocess timeout is not a security sandbox and is for trusted local candidates only. External agent trials have **not** been performed in the included experiment.

