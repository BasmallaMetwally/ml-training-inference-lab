import math


def fit(rows, weights):
    try:
        valid = len(rows) == len(weights) and all(
            math.isfinite(w) and w >= 0 for w in weights
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
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
        largest_weight = max(w for _, w in pairs)
        relative = [w / largest_weight for _, w in pairs]
        total = math.fsum(relative)
        fractions = [w / total for w in relative]
        values = [x for x, _ in pairs]
        if all(x == values[0] for x in values):
            state.append([values[0], 1.0])
            continue
        mean = math.fsum(x * fraction for x, fraction in zip(values, fractions))
        differences = [x - mean for x in values]
        if all(math.isfinite(d) for d in differences):
            scale = math.hypot(*(math.sqrt(f) * d
                                 for f, d in zip(fractions, differences)))
        else:
            magnitude = max(abs(x) for x in values)
            scaled = [x / magnitude for x in values]
            center = math.fsum(x * f for x, f in zip(scaled, fractions))
            mean = center * magnitude
            scale = magnitude * math.hypot(*(
                math.sqrt(f) * (x - center)
                for x, f in zip(scaled, fractions)
            ))
        state.append([mean, scale if scale > 0 else 1.0])
    return state


def transform(rows, state):
    return [[None if x is None else (x - state[j][0]) / state[j][1]
             for j, x in enumerate(row)] for row in rows]
