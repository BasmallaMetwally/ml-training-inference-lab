from core import rank, ndcg


def solve(p):
    ids = rank(p['items'])
    k = p['k']
    return {'ids': ids[:max(k, 0)], 'ndcg': ndcg(ids, p['judgments'], k)}
