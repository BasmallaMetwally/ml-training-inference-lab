# early_stop_resume

core.advance(state,value,index,patience,delta) tracks strict improvements value < best-delta. State keys best (null means +infinity), best_index, stale, seen, stopped. Finite values only; NaN/infinity raise ValueError. Patience>=1 and delta>=0 else ValueError. seen increments per consumed value; stop when stale>=patience. pipeline.solve accepts state or null (initial), values,patience,delta; indices continue from state's seen, must not consume values after stopped. Return resulting state; never mutate supplied state. Empty values still validate patience/delta.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
