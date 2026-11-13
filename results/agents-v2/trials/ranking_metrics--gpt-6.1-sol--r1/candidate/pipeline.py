from core import auc, ap


def solve(p):
    y, s = p['y'], p['score']
    w = p.get('w', p.get('weights', [1.0] * len(y)))
    return {'auc': auc(y, s, w), 'ap': ap(y, s, w)}
