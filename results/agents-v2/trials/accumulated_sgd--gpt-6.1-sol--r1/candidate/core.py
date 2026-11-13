import math


def gradient(weights, x, y):
    if not x:
        return [0.0] * len(weights)
    result = [0.0] * len(weights)
    for row, target in zip(x, y):
        error = sum(a * b for a, b in zip(row, weights)) - target
        for j in range(len(weights)):
            result[j] += error * row[j]
    return [2.0 * (value / len(x)) for value in result]


def step(weights, velocity, g, lr, momentum, clip):
    if clip < 0:
        raise ValueError('clip')
    raw = list(g)
    if clip == 0:
        clipped = [0.0] * len(raw)
    else:
        largest = max((abs(value) for value in raw), default=0.0)
        if largest:
            unit_norm = math.sqrt(sum((value / largest) ** 2 for value in raw))
            if largest > clip / unit_norm:
                clipped = [(value / largest) * (clip / unit_norm) for value in raw]
            else:
                clipped = raw
        else:
            clipped = raw
    new_velocity = [momentum * v + d for v, d in zip(velocity, clipped)]
    new_weights = [w - lr * v for w, v in zip(weights, new_velocity)]
    return new_weights, new_velocity
