# adamw_resume

core.update(w,m,v,g,t,lr,b1,b2,eps,decay) implements AdamW: new moments, bias correction at incremented step t, epsilon outside sqrt, decoupled decay on old weights; do not decay moments. pipeline.solve starts from supplied weights,m,v,step, consumes gradients in sequence, increments step before each update and returns weights,m,v,step. Empty gradient sequence must return unchanged state. Lists only, no input mutation.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
