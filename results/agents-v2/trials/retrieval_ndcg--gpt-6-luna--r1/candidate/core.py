import math


def rank(items):
    seen = set()
    for record in items:
        doc_id = record['id']
        if doc_id in seen:
            raise ValueError('duplicate id')
        seen.add(doc_id)
    return [record['id'] for record in sorted(items, key=lambda record: (-record['score'], record['id']))]


def ndcg(ids, judgments, k):
    if k <= 0:
        return 0.0

    def gain(relevance):
        return 2 ** relevance - 1

    dcg = sum(
        gain(judgments.get(doc_id, 0)) / math.log2(position + 2)
        for position, doc_id in enumerate(ids[:k])
    )
    ideal_relevances = sorted(judgments.values(), reverse=True)[:k]
    ideal_dcg = sum(
        gain(relevance) / math.log2(position + 2)
        for position, relevance in enumerate(ideal_relevances)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0
