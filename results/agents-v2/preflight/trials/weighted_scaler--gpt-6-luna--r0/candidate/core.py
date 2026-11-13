import math


def fit(rows, weights):
    try:
        if len(rows) != len(weights):
            raise ValueError('weights')
        checked_weights = []
        for weight in weights:
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('weights')
            checked_weights.append(weight)
    except (TypeError, OverflowError):
        raise ValueError('weights')

    width = len(rows[0]) if rows else 0
    state = []
    for j in range(width):
        pairs = [(row[j], weight) for row, weight in zip(rows, checked_weights)
                 if row[j] is not None and weight > 0]
        total = sum(weight for _, weight in pairs)
        if total == 0:
            state.append([0.0, 1.0])
            continue
        mean = sum(value * weight for value, weight in pairs) / total
        variance = sum(weight * (value - mean) ** 2 for value, weight in pairs) / total
        scale = math.sqrt(variance) if variance > 0 else 1.0
        state.append([mean, scale])
    return state


def transform(rows, state):
    return [[None if value is None else (value - state[j][0]) / state[j][1]
             for j, value in enumerate(row)] for row in rows]
