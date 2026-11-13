from core import auc, ap


def solve(p):
    y, s = p['y'], p['score']
    if 'w' in p:
        w = p['w']
    elif 'weights' in p:
        w = p['weights']
    else:
        w = [1.0] * len(y)
    return {'auc': auc(y, s, w), 'ap': ap(y, s, w)}
