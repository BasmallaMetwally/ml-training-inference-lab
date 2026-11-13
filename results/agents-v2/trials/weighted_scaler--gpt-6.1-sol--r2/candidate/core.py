import math


def fit(rows, weights):
    if len(rows) != len(weights):
        raise ValueError('weights')
    for weight in weights:
        try:
            valid = math.isfinite(weight) and weight >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError('weights')

    width = len(rows[0]) if rows else 0
    state = []
    for j in range(width):
        pairs = [(row[j], weight) for row, weight in zip(rows, weights)
                 if row[j] is not None and weight > 0]
        if not pairs:
            state.append([0.0, 1.0])
            continue
        # Normalize weights before summing, so large finite weights are safe.
        largest = max(weight for _, weight in pairs)
        relative = [(x, weight / largest) for x, weight in pairs]
        total = math.fsum(weight for _, weight in relative)
        if all(x == pairs[0][0] for x, _ in pairs):
            state.append([pairs[0][0], 1.0])
            continue
        mean = math.fsum(x * (weight / total) for x, weight in relative)
        # hypot computes the population standard deviation without squaring
        # large deviations or underflowing small squared deviations.
        scale = math.hypot(*(math.sqrt(weight / total) * (x - mean)
                             for x, weight in relative))
        state.append([mean, scale if scale > 0 else 1.0])
    return state


def transform(rows, state):
    return [[None if x is None else (x - state[j][0]) / state[j][1]
             for j, x in enumerate(row)] for row in rows]
