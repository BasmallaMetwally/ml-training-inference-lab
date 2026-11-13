# ML Training and Inference Lab

## Measured experiment

Fashion-MNIST classification with a small PyTorch MLP, trained from random initialization on CPU.

- Train / validation / test: 50,000 / 10,000 / 10,000 images.
- Seed: 42; epochs: 10; selected epoch: 9 (minimum validation loss).
- Official test accuracy: **88.79%**; test cross entropy: **0.3258**.
- Majority-class baseline: 10.00%.
- Training wall time (epoch loops): 4.4 seconds.
- Environment: Python 3.12.14, PyTorch 2.15.0, arm64, 1 CPU thread(s).
- Checkpoint SHA256: `dfac5883e6adc56dd89372b515e607df9658c658b878859582d84125b5a86771`.

## Training history

| Epoch | Train loss | Validation loss | Validation accuracy | Seconds |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6179 | 0.3978 | 85.59% | 0.46 |
| 2 | 0.4114 | 0.3665 | 86.70% | 0.44 |
| 3 | 0.3714 | 0.3563 | 87.06% | 0.44 |
| 4 | 0.3504 | 0.3336 | 88.02% | 0.44 |
| 5 | 0.3301 | 0.3267 | 88.21% | 0.45 |
| 6 | 0.3157 | 0.3276 | 88.15% | 0.45 |
| 7 | 0.2997 | 0.3143 | 88.90% | 0.44 |
| 8 | 0.2906 | 0.3197 | 88.58% | 0.44 |
| 9 | 0.2847 | 0.3078 | 88.95% | 0.44 |
| 10 | 0.2741 | 0.3090 | 89.03% | 0.44 |

## CPU inference benchmark

Same selected weights, eval mode, preloaded normalized validation images, warm-up, randomized interleaved repeats and fixed thread count. Times include input validation. Batch latency is not request latency.

| Variant | Batch | Median batch ms | Repeat range ms | Images/s | vs autograd | vs inference mode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eval_autograd | 1 | 0.0544 | 0.0537–0.0549 | 18,381 | 1.00× | 0.81× |
| eval_autograd | 32 | 0.0969 | 0.0964–0.0975 | 330,269 | 1.00× | 0.85× |
| eval_autograd | 128 | 0.1963 | 0.1959–0.2001 | 651,961 | 1.00× | 0.94× |
| eval_autograd | 512 | 0.6064 | 0.6026–0.6082 | 844,345 | 1.00× | 0.98× |
| eval_inference_mode | 1 | 0.0442 | 0.0430–0.0445 | 22,646 | 1.23× | 1.00× |
| eval_inference_mode | 32 | 0.0819 | 0.0814–0.0825 | 390,577 | 1.18× | 1.00× |
| eval_inference_mode | 128 | 0.1851 | 0.1845–0.1862 | 691,482 | 1.06× | 1.00× |
| eval_inference_mode | 512 | 0.5956 | 0.5934–0.5979 | 859,643 | 1.02× | 1.00× |
| fused_inference_mode | 1 | 0.0319 | 0.0317–0.0321 | 31,361 | 1.71× | 1.38× |
| fused_inference_mode | 32 | 0.0690 | 0.0688–0.0692 | 463,746 | 1.40× | 1.19× |
| fused_inference_mode | 128 | 0.1689 | 0.1686–0.1710 | 757,683 | 1.16× | 1.10× |
| fused_inference_mode | 512 | 0.5579 | 0.5568–0.5601 | 917,655 | 1.09× | 1.07× |

## Correctness of fusion

Folded two Linear/BatchNorm pairs using running statistics. Maximum absolute logit difference: 0.00001621; prediction disagreements: 0/10000.
Validation accuracy: 88.95% before and 88.95% after fusion.

## Limits

One seed and one CPU machine; no GPU, production-load, or cross-machine performance claims. The test split is used after checkpoint selection; optimization equivalence is checked on validation data. This is a small educational dataset, not evidence of production model quality. Timing excludes preprocessing and I/O. Parameter storage is reported separately from peak memory, which is not measured. Debugging-task validation is not an observed external coding-agent pass rate.

Full configuration, epoch metrics, confusion matrix, environment and raw timing blocks accompany this report.
