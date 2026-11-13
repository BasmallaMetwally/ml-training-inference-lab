import math
from numbers import Integral


def validate(rows, y, w, n):
    try:
        if not isinstance(n, Integral) or n < 1:
            raise ValueError('n must be a positive integer')
        if len(rows) != len(y) or len(y) != len(w):
            raise ValueError('matching lengths required')
        if any(not math.isfinite(v) or v < 0 for v in w):
            raise ValueError('invalid weights')
        for r, t in zip(rows, y):
            if not r or not isinstance(t, Integral) or not 0 <= t < len(r):
                raise ValueError('invalid label or empty probabilities')
            if any(not math.isfinite(v) or v < 0 or v > 1 for v in r):
                raise ValueError('invalid probabilities')
            if abs(math.fsum(r) - 1.0) > 1e-8:
                raise ValueError('probabilities must sum to one')
    except (TypeError, OverflowError) as exc:
        raise ValueError('invalid inputs') from exc


def bins(rows, y, w, n):
    validate(rows, y, w, n)
    accum = [[0.0, 0.0, 0.0] for _ in range(n)]
    for r, t, v in zip(rows, y, w):
        pred = max(range(len(r)), key=lambda i: r[i])
        conf = r[pred]
        slot = min(int(conf * n), n - 1)
        accum[slot][0] += v
        accum[slot][1] += v * (pred == t)
        accum[slot][2] += v * conf
    return [
        {'weight': a, 'accuracy': b / a if a else 0.0,
         'confidence': c / a if a else 0.0}
        for a, b, c in accum
    ]


def brier(rows, y, w):
    validate(rows, y, w, 1)
    total = sum(w)
    if not total:
        return 0.0
    return sum(
        v * math.fsum((a - (j == t)) ** 2 for j, a in enumerate(r))
        for r, t, v in zip(rows, y, w)
    ) / total
