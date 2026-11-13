# accumulated_sgd

Pure-list linear regression optimizer, no bias. core.gradient(weights,x,y) returns gradient of mean squared error (factor 2). Empty batch contributes no samples. core.step weights, velocity, gradient, lr,momentum,clip: clip raw gradient to global L2 norm at most clip before velocity=momentum*velocity+gradient, then weights-=lr*velocity; clip=0 means zero gradient. solve processes payload steps (each list of batches {x,y}) with sample-weighted accumulation, one update per nonempty step; empty step leaves state unchanged. Return weights and velocity, preserve all inputs. Invalid negative clip raises ValueError even for empty steps.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
