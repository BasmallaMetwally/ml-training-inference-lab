from core import auc, ap


def solve(p):
    y, score = p['y'], p['score']
    weights = p.get('weights')
    if weights is None:
        weights = [1.0] * len(y)
    return {'auc': auc(y, score, weights), 'ap': ap(y, score, weights)}
