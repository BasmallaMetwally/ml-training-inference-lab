import math


def rank(items):
    ids = [record['id'] for record in items]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate id')
    return [record['id'] for record in sorted(items, key=lambda record: (-record['score'], record['id']))]


def ndcg(ids, judgments, k):
    if k <= 0:
        return 0.0

    cutoff = int(k)
    dcg = sum(
        (2 ** judgments.get(doc, 0) - 1) / math.log2(rank_index + 2)
        for rank_index, doc in enumerate(ids[:cutoff])
    )
    ideal_relevances = sorted(judgments.values(), reverse=True)[:cutoff]
    ideal_dcg = sum(
        (2 ** relevance - 1) / math.log2(rank_index + 2)
        for rank_index, relevance in enumerate(ideal_relevances)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0
