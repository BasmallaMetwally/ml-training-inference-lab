# ranking_metrics

Weighted binary ranking. core.auc(y,score,w): probability a positive outranks a negative, ties count half, weighted pairs; return None if either class has zero weight. core.ap uses threshold-grouped average precision: descending equal-score groups included together, precision at end of group times recall increment; None if no positive weight. solve returns {auc, ap}, uses payload weights, or ones when absent. Nonbinary labels, negative weights, unequal lengths raise ValueError. Empty input is valid.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
