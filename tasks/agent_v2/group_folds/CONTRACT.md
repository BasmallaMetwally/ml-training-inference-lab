# group_folds

Deterministic group-exclusive folds. core.assign(groups,k,seed): reject k<2; count each string group; visit groups sorted by (-count, sha256((str(seed)+':'+group).encode()).hexdigest(),group). Greedily put group in fold with smallest current number of rows, tie smallest fold index. solve returns list of k lists of original row indices in input order, using payload k and seed, including empty folds. No reliance on global RNG, built-in randomized hash, or lexical group order alone.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
