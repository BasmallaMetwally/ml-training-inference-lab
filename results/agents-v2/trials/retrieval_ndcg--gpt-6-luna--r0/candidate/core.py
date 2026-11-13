import math


def rank(items):
    ids = [item['id'] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate id')
    return [item['id'] for item in sorted(items, key=lambda item: (-item['score'], item['id']))]


def ndcg(ids, judgments, k):
    if k <= 0:
        return 0.0

    dcg = sum(
        (2 ** judgments.get(doc, 0) - 1) / math.log2(rank + 2)
        for rank, doc in enumerate(ids[:k])
    )
    ideal_relevances = sorted(judgments.values(), reverse=True)[:k]
    ideal = sum(
        (2 ** relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(ideal_relevances)
    )
    return dcg / ideal if ideal != 0 else 0.0
