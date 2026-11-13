# time_windows

Records {entity:str,t:int,value:number}. core.windows sorts by entity then t, rejects duplicate (entity,t), and uses consecutive timestamps only. For each entity and each start index 0,stride,2*stride,... form lookback feature values followed by horizon targets; include only if timestamps across full window increase by exactly 1. Require lookback,horizon,stride>=1. pipeline.solve splits RAW records at t<cutoff vs t>=cutoff BEFORE constructing windows; no train history in eval windows. Return {train,eval} each list of {entity,start,x,y}; start is first target timestamp. Sort output entity then start. Empty partition allowed; do not mutate rows.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
