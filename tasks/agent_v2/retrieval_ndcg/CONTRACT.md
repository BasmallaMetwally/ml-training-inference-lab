# retrieval_ndcg

core.rank(items): records {id:str,score:number}, sort descending score, ties ascending id; duplicate id raises ValueError. core.ndcg(ids,judgments,k) uses gain 2**relevance-1, log2(rank+2), absent judgment relevance zero; ideal DCG from ALL judged documents, descending relevance, at most k. k<=0 gives 0. Zero ideal gives 0. pipeline.solve respects supplied k (not candidate count), returns ranked IDs limited max(k,0), and ndcg. IDs not retrieved can affect ideal denominator.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
