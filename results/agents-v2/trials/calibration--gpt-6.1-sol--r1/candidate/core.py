import math
import operator


def validate(rows, y, w, n):
    try:
        operator.index(n)
        if n < 1 or len(rows) != len(y) or len(y) != len(w):
            raise ValueError('inputs')
        if any(not math.isfinite(v) or v < 0 for v in w):
            raise ValueError('weights')
        for r, t in zip(rows, y):
            operator.index(t)
            if (not r or not 0 <= t < len(r)
                    or any(not math.isfinite(v) or v < 0 or v > 1 for v in r)
                    or abs(math.fsum(r) - 1.0) > 1e-8):
                raise ValueError('probabilities')
    except (TypeError, OverflowError) as exc:
        raise ValueError('inputs') from exc


def bins(rows, y, w, n):
    validate(rows, y, w, n)
    accum = [[0.0, 0.0, 0.0] for _ in range(n)]
    for r, t, v in zip(rows, y, w):
        pred = max(range(len(r)), key=lambda i: r[i])
        conf = r[pred]
        slot = min(int(conf * n), n - 1)
        # Compare against the boundaries themselves to avoid multiplication
        # rounding an exact boundary into its neighboring bin.
        if slot > 0 and conf < slot / n:
            slot -= 1
        elif slot < n - 1 and conf >= (slot + 1) / n:
            slot += 1
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
