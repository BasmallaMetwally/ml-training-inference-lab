import math


def validate(rows, y, weights, n):
    try:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError('inputs')
        if len(rows) != len(y) or len(y) != len(weights):
            raise ValueError('inputs')
    except (TypeError, OverflowError):
        raise ValueError('inputs') from None

    for weight in weights:
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError('weights')
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('weights')

    for row, label in zip(rows, y):
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise ValueError('probabilities')
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(row):
            raise ValueError('labels')
        for probability in row:
            if (not isinstance(probability, (int, float)) or isinstance(probability, bool)
                    or not math.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError('probabilities')
        if abs(sum(row) - 1.0) > 1e-8:
            raise ValueError('probabilities')


def bins(rows, y, weights, n):
    validate(rows, y, weights, n)
    accum = [[0.0, 0.0, 0.0] for _ in range(n)]
    for row, label, weight in zip(rows, y, weights):
        prediction = max(range(len(row)), key=lambda i: row[i])
        confidence = row[prediction]
        slot = min(int(confidence * n), n - 1)
        accum[slot][0] += weight
        accum[slot][1] += weight * (prediction == label)
        accum[slot][2] += weight * confidence
    return [
        {'weight': total, 'accuracy': correct / total if total else 0.0,
         'confidence': confidence / total if total else 0.0}
        for total, correct, confidence in accum
    ]


def brier(rows, y, weights):
    validate(rows, y, weights, 1)
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(
        weight * sum((probability - (class_index == label)) ** 2
                     for class_index, probability in enumerate(row))
        for row, label, weight in zip(rows, y, weights)
    ) / total
