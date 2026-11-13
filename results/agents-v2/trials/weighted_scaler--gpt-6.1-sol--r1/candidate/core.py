import math


def fit(rows, weights):
    if len(rows) != len(weights):
        raise ValueError('weights')
    try:
        valid = all(math.isfinite(w) and w >= 0 for w in weights)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('weights') from None
    if not valid:
        raise ValueError('weights')

    width = len(rows[0]) if rows else 0
    state = []
    for j in range(width):
        pairs = [(row[j], w) for row, w in zip(rows, weights)
                 if row[j] is not None and w > 0]
        if not pairs:
            state.append([0.0, 1.0])
            continue
        if all(x == pairs[0][0] for x, _ in pairs):
            state.append([pairs[0][0], 1.0])
            continue
        # Normalize weights before summing so finite weights cannot overflow.
        largest = max(w for _, w in pairs)
        normalized = [(x, w / largest) for x, w in pairs]
        total = math.fsum(w for _, w in normalized)
        mean = math.fsum(x * (w / total) for x, w in normalized)
        # The norm avoids overflow from squaring large finite deviations.
        scale = math.hypot(*(math.sqrt(w / total) * (x - mean)
                             for x, w in normalized))
        state.append([mean, scale if scale > 0 else 1.0])
    return state


def transform(rows, state):
    return [[None if x is None else (x - state[j][0]) / state[j][1]
             for j, x in enumerate(row)] for row in rows]
