import math


def rank(items):
    if len({record['id'] for record in items}) != len(items):
        raise ValueError('duplicate')
    return [record['id'] for record in sorted(items, key=lambda record: (-record['score'], record['id']))]


def ndcg(ids, judgments, k):
    if k <= 0:
        return 0.0
    dcg = sum(
        (2 ** judgments.get(doc, 0) - 1) / math.log2(i + 2)
        for i, doc in enumerate(ids[:k])
    )
    ideal = sum(
        (2 ** relevance - 1) / math.log2(i + 2)
        for i, relevance in enumerate(sorted(judgments.values(), reverse=True)[:k])
    )
    return dcg / ideal if ideal else 0.0
