# What a small ML engineering study actually showed

I built an ML training and inference lab with coding-agent assistance, then tested its claims rather than stopping at a demo.

Across ten paired seeds, a small CNN improved Fashion-MNIST accuracy by 0.54 percentage points over an MLP, with a 95% seed-level interval of 0.44–0.64. It used about twelve times fewer parameters but had higher batch-1 CPU latency on the measured machine.

I also compared checked and unchecked inference, profiled operator calls, measured peak process RSS, and tested fusion, compilation, dynamic int8 and ONNX. Optimizations that changed predictions were reported as rejected under an exact-parity policy.

For evaluation practice, two models each attempted twelve ML repair tasks three times. Both passed every task under withheld Docker grading. That is a ceiling effect, not a credible model ranking. Transcripts, uncertainty caveats and the negative findings are included.

Finally, checkpoint resumption was bitwise reproducible within Linux and macOS, while the cross-platform tensors differed.

Code and evidence: https://github.com/BasmallaMetwally/ml-training-inference-lab

Archived v1.0: https://doi.org/10.5281/zenodo.24201648
