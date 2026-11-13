# Model card

**Version:** v1.0. **Author:** Basmala Metwally. **Purpose:** a reproducible ML engineering study and coding-agent repair pilot. The code was developed with coding-agent assistance.

**Models:** an MLP with 109,770 trainable parameters and a small CNN with 9,098. Both consume flattened, normalized 28×28 grayscale Fashion-MNIST images and return ten class logits. Adam, learning rate 0.001, batch size 128, ten CPU epochs. Normalization is fitted on 50,000 training images; a fixed 10,000-image validation split selects minimum-loss checkpoints.

**Evaluation:** ten initialization/shuffle seeds per architecture with a fixed split. Official test accuracy: MLP **88.428% ± 0.103 percentage points** and CNN **88.968% ± 0.202 percentage points** (sample standard deviation). The paired CNN improvement is **0.540 percentage points**, 95% seed-level t interval **0.435–0.645**, exact two-sided sign-flip p **0.001953**. These intervals describe training-seed variability conditional on this dataset split.

**Inference:** eval and inference mode; checked and unchecked ablations, Linear/BatchNorm fusion, torch.compile, dynamic int8 and ONNX Runtime. A runtime is eligible for parity-preserving speed claims only with zero validation class prediction disagreements. Results, setup costs and platform failures are explicitly recorded. Latency is in-memory CPU forward time; peak RSS includes runtime/import overhead, not just activations.

**Intended use:** inspecting training, restoration, numerical correctness, performance tradeoffs and evaluation methodology. The models are educational garment classifiers, not general visual recognition systems or production service benchmarks.

**Limitations:** one dataset, fixed split, ten seeds and a small architecture comparison. Shirt/T-shirt and other visually similar garment classes remain error sources. No calibration, out-of-distribution robustness, demographic analysis, GPU or distributed-training claims. There is no evidence here of professional deployment experience.

**Artifacts and license:** source under MIT; selected checkpoints and exported/runtime artifacts are separate GitHub Release assets. SHA256 provenance links checkpoints to exported graphs. Dataset attribution is in DATA_SHEET.md and THIRD_PARTY.md. Code/model versioning does not imply third-party certification.

## Post-v1.0 corrections and exploratory extension

The original Linux peak-memory comparison is inconclusive; use the [replacement VmHWM study](results/performance-linux/corrected/REPORT.md). The [six-capacity, three-seed sweep](results/capacity-sweep/REPORT.md) is exploratory test-set reuse. It does not replace the original ten-seed accuracy comparison. Architecture seed blocks share shuffle order, but not initialization or dropout; pairing gives no guaranteed variance reduction. An unpaired Welch sensitivity check is recorded in results/study/summary.json.

V2 coding-agent evaluation is a separate synthetic software-repair experiment; see [its report](results/agents-v2/REPORT.md). Its scores are not Fashion-MNIST model performance or a general model ranking.
