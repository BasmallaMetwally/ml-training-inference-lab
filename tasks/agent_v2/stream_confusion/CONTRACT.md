# stream_confusion

Payload shards is a list of {actual:[str], predicted:[str]}, labels is a list of strings. Build confusion counts with rows=actual, columns=predicted. core.count accumulates each shard without mutating it; core.merge aligns matrices by label strings, union sorted lexically. Include initial supplied labels even unseen; reject duplicated initial labels or unequal actual/predicted lengths with ValueError. solve returns labels, matrix, macro_f1 averaged over every union label, treating zero denominator as zero. Empty union gives macro_f1=0.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
