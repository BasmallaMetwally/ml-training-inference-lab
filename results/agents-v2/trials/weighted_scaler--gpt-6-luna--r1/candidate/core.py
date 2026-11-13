import math


def fit(rows, weights):
    try:
        valid_weights = len(rows) == len(weights) and all(
            math.isfinite(w) and w >= 0 for w in weights
        )
    except (TypeError, ValueError, OverflowError):
        valid_weights = False
    if not valid_weights:
        raise ValueError('weights')

    width = len(rows[0]) if rows else 0
    state = []
    for j in range(width):
        pairs = [(row[j], w) for row, w in zip(rows, weights) if row[j] is not None]
        total = sum(w for _, w in pairs)
        if total <= 0:
            state.append([0.0, 1.0])
            continue
        mean = sum(x * w for x, w in pairs) / total
        variance = sum(w * (x - mean) ** 2 for x, w in pairs) / total
        scale = math.sqrt(variance) if variance > 0 else 1.0
        state.append([mean, scale])
    return state


def transform(rows, state):
    return [
        [None if x is None else (x - state[j][0]) / state[j][1] for j, x in enumerate(row)]
        for row in rows
    ]
