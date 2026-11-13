import bisect
import math

def validate(rows, y, w, n=None):
    try:
        if (len(rows) != len(y) or len(y) != len(w) or
                (n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1))):
            raise ValueError('inputs')
        for weight in w:
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('inputs')
        for row, label in zip(rows, y):
            if not row or not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(row):
                raise ValueError('probabilities')
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in row):
                raise ValueError('probabilities')
            if abs(sum(row) - 1) > 1e-8:
                raise ValueError('probabilities')
    except (TypeError, OverflowError):
        raise ValueError('inputs')

def bins(rows, y, weights, n):
    validate(rows, y, weights, n)
    accum = [[0.0, 0.0, 0.0] for _ in range(n)]
    boundaries = [i / n for i in range(1, n)]
    for row, label, weight in zip(rows, y, weights):
        pred = max(range(len(row)), key=lambda i: row[i])
        confidence = row[pred]
        slot = bisect.bisect_right(boundaries, confidence)
        accum[slot][0] += weight
        accum[slot][1] += weight * (pred == label)
        accum[slot][2] += weight * confidence
    return [{'weight': total, 'accuracy': correct / total if total else 0.0,
             'confidence': conf / total if total else 0.0}
            for total, correct, conf in accum]

def brier(rows, y, weights):
    validate(rows, y, weights)
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(weight * sum((value - (index == label)) ** 2
                            for index, value in enumerate(row))
               for row, label, weight in zip(rows, y, weights)) / total
