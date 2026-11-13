# ML Training and Inference Lab

A compact ML engineering project: train a neural classifier from scratch, select a checkpoint using validation loss, evaluate a held-out test split, and measure CPU inference improvements without changing predicted classes.

The project includes a paired 10-seed MLP/CNN study, error analysis, and twelve ML repair tasks. Six actual model responses (two models, three repetitions) are recorded with transcripts; their withheld Docker grading is being run separately. Three older public development fixtures remain available as examples.

See [10-seed study](results/study/summary.json), [Pareto plot](results/study/pareto.png), [error examples](results/study/shirt_errors.png), and [agent protocol](results/agents/protocol.json).

## Recorded result

- **88.79% test accuracy** on all 10,000 official Fashion-MNIST test images.
- **50,000 / 10,000 / 10,000** train / validation / test images; stratified split, seed 42.
- Ten CPU training epochs; checkpoint selected at **epoch 9 by minimum validation cross entropy**, not by test accuracy.
- Two Linear/BatchNorm pairs fused for inference; **zero class prediction changes** across 10,000 validation images.
- At batch size 1, fusion gave **1.38× speedup over unfused inference mode** in the recorded CPU experiment (0.0442 ms to 0.0319 ms per forward); gains were smaller at larger batches. These are machine-specific forward timings.
- Full-run resumption verified: **10 uninterrupted epochs and 5 + restore + 5 produced bitwise-identical model tensors** on the recorded CPU environment.
- **31 offline tests passed** locally.
- Three reference repair tasks pass; all seven deliberately wrong fixtures are rejected. These are fixture results, not agent success rates.

See [experiment report](results/seed42/REPORT.md), [standalone HTML report](results/seed42/report.html), [raw measurements](results/seed42/benchmark.json), [resume verification](results/seed42/resume_verification.json) and [task validation](results/task_validation.json). Exact test counts are recorded in [test results](results/tests.xml).

## Quick start

Python 3.11+ is required. The recorded experiment used Python 3.12.14, PyTorch 2.15.0 and NumPy 2.5.4 on macOS arm64, using one CPU thread. No GPU or API key is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m pytest -q
ml-lab download
ml-lab train --epochs 10 --run runs/reproduce
ml-lab evaluate --run runs/reproduce
ml-lab benchmark --run runs/reproduce
ml-lab report --run runs/reproduce --out results/reproduced
```

Open `results/reproduced/report.html` for the report. Dataset download requires internet; tests use synthetic fixtures and run offline. The data files are verified against the original published MD5 values and recorded with SHA256 hashes.

For exact package versions in the recorded environment, use `requirements-lock.txt`, then `python -m pip install --no-deps --no-build-isolation -e .`. Hardware and library differences may still change timing or floating-point results. The lockfile describes the tested macOS environment, not a universal cross-platform binary lock.

The downloadable source archive includes `runs/seed42/best.pt` and `last.pt`; raw dataset files are excluded and can be downloaded using the command above. Training checkpoints and downloaded data are excluded from Git by default.

## Model and training

```text
784 input pixels
  -> Linear 128 -> BatchNorm -> ReLU -> Dropout 0.2
  -> Linear 64  -> BatchNorm -> ReLU -> Dropout 0.2
  -> Linear 10 logits
```

Training uses Adam at learning rate 0.001, batch size 128 and mean cross entropy. Gradient norms are checked for finite values and clipped at 5. The loss consumes logits directly. Input shape and finiteness checks prevent malformed inputs from silently propagating.

Normalization mean and standard deviation are fitted on the training subset only. Validation is used for checkpoint selection; the official test split is read only during the explicit `evaluate` stage. There was no test-driven hyperparameter search in the recorded experiment. The 10% majority-class baseline is a sanity check, not a competitive model benchmark.

## Checkpointing and reproducibility

Each checkpoint includes model and optimizer state, CPU RNG state (used by dropout), an independent shuffle-generator state, the completed epoch, split-index hashes, normalization, configuration and epoch history. Writes use a temporary file followed by replacement. Resume rejects changed data/split metadata or training configuration and retains the earlier best checkpoint.

```bash
ml-lab train --run runs/resume-demo --epochs 5
ml-lab train --run runs/resume-demo --epochs 10 --resume
```

`--epochs` is the total target epoch count, not the number of additional epochs. Exact resumption is checked in unit/integration tests and in the recorded full-dataset experiment. To repeat that full-dataset verification from a fresh source checkout:

```bash
ml-lab download
ml-lab train --epochs 10
python scripts/verify_experiment.py
```

Use a fresh workspace: existing runs are protected from accidental overwrite. Exact equality is a same-environment CPU result, not a guarantee across PyTorch versions or devices.

## Inference performance experiment

Three variants use the same trained weights and eval mode:

1. Eval forward with autograd enabled, as a diagnostic baseline.
2. Eval forward under `torch.inference_mode()`.
3. Eval forward with Linear/BatchNorm fusion under `torch.inference_mode()`.

The report includes comparisons against **both** baselines so the effect of fusion can be separated from disabling autograd. Dropout is disabled in all three; model quality is not traded for speed.

Fusion folds BatchNorm running statistics into Linear weights and biases. Before timing, the code compares logits with explicit tolerances and requires identical class predictions over the full validation split. It also checks validation accuracy before and after fusion.

Timing uses `torch.utils.benchmark.Timer.blocked_autorange`, 20 warm-up forwards, fixed thread count and randomly interleaved repeated trials at batch sizes 1, 32, 128 and 512. Raw timing blocks and per-repeat spread are saved. Data preparation, disk I/O, checkpoint loading and context entry are excluded; forward input validation is included. Results are local steady-state forward timings, not API request latency or service capacity. No speedup is assumed in advance.

Parameter tensor storage is reported, **not peak RAM**. There are no GPU, quantization or production-serving claims.

## Debugging and coding-agent tasks

See [task contracts and evaluation protocol](tasks/README.md). To demonstrate grader behavior:

```bash
ml-lab grade --task stable_loss --candidate tasks/reference.py
ml-lab grade --task stable_loss --candidate tasks/starter.py
python scripts/validate_tasks.py
```

The second command deliberately exits with code 1. Covered failures include missing optimizer updates, stale gradients, unstable exponential normalization, detached loss gradients, active dropout/BatchNorm during inference and retained autograd graphs. The local timeout runner distinguishes wrong output, crash and timeout. It does not sandbox untrusted code.

The public legacy fixtures are separate from the twelve withheld tasks. Agent generation records are under `results/agents`; withheld grading and statistical summaries are produced separately. No model ranking is inferred from generation alone.

## Validation and CI

Offline tests cover:

- Learning a known synthetic decision boundary.
- Disjoint, stratified, reproducible data splits and corrupt IDX rejection.
- Invalid shapes, NaN inputs and non-finite gradients.
- Model and optimizer restoration, dropout RNG, shuffle RNG and exact full-workflow resumption.
- Safe evaluation and fusion with nontrivial BatchNorm running statistics.
- Reference/starter/mutant grading, timeouts and crashes.

A GitHub Actions workflow tests Python 3.12 on Linux and macOS and records same-platform resumption and cross-platform tensor fingerprints. Hosted results will be linked after the first completed run; no unverified CI badge is shown.

## Repository map

```text
src/ml_lab/       data, model, training, benchmark, report and CLI
tests/            offline correctness and workflow tests
tasks/            starter, references and hand-authored incorrect fixtures
scripts/          full-dataset resume verification and task validation
results/          measured metrics, report, raw timing and validation evidence
.github/workflows/ci.yml
requirements-lock.txt
```

## Limits and next experiments

This is a small MLP experiment on an educational image dataset, using one seed and one CPU environment. It demonstrates ML engineering workflows; it does not establish state-of-the-art classification, distributed training, memory optimization, or professional production experience. Validation data are reused for checkpoint selection and fusion verification, so the fusion correctness check is not a separate generalization benchmark.

Useful next experiments are multiple training seeds, a CNN baseline, separate peak-memory profiling, and actual coding-agent repair trials with recorded transcripts. Those are not implemented or claimed here.

## Sources

- [Fashion-MNIST official repository, files, checksums and class definitions](https://github.com/zalandoresearch/fashion-mnist).
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
- [PyTorch benchmark tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).
- [Linear/BatchNorm fusion API](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.fusion.fuse_linear_bn_eval.html).

Source code: MIT license. Dataset attribution and licensing: [THIRD_PARTY.md](THIRD_PARTY.md). Arabic walkthrough: [GUIDE.ar.md](GUIDE.ar.md).

