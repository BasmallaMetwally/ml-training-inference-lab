# weighted_scaler

Implement fit(rows, weights) and transform(rows, state) in core.py and solve in pipeline.py.
Payload: train, weights, other. Output transformed other rows. Fit each feature using only non-None training values and corresponding nonnegative weights; population weighted variance. A feature with no positive observed weight uses mean=0, scale=1. Constant features use scale=1. None stays None. Never mutate inputs. Weights must match train length, be finite and nonnegative; invalid weights raise ValueError. Empty train returns [] only when other is also empty; otherwise infer width from other with default state.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
