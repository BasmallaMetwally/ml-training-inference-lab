# calibration

Multiclass probabilities. core.bins(rows,y,weights,n): equal-width confidence bins [i/n,(i+1)/n), last includes 1. Top-class ties choose lowest class index. Return length-n records {weight,accuracy,confidence}, empty bins all zero. core.brier is weighted mean SUM squared class errors (not divided by classes). solve returns {bins,ece,brier}, ECE weighted absolute bin accuracy-confidence differences. Zero total weight yields zero ECE/Brier. n>=1, matching lengths, nonnegative weights, nonempty class vectors, labels in range, finite probabilities in [0,1] summing to 1 within 1e-8 required, else ValueError.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
