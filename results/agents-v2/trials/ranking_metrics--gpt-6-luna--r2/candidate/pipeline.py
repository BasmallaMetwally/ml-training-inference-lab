from core import auc, ap


def solve(p):
    y, scores = p['y'], p['score']
    weights = p.get('weight', p.get('weights'))
    if weights is None:
        weights = [1.0] * len(y)
    return {'auc': auc(y, scores, weights), 'ap': ap(y, scores, weights)}
