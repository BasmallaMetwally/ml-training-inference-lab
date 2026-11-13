import math


def gradient(weights, x, y):
    if not x:
        return [0.0] * len(weights)
    residuals = [sum(a * b for a, b in zip(row, weights)) - target
                 for row, target in zip(x, y)]
    return [2.0 * sum(error * row[j] for error, row in zip(residuals, x)) / len(x)
            for j in range(len(weights))]


def step(weights, velocity, g, lr, momentum, clip):
    if clip < 0:
        raise ValueError('clip')
    norm = math.hypot(*g)
    if clip == 0:
        clipped = [0.0] * len(g)
    elif norm > clip:
        clipped = [d * (clip / norm) for d in g]
    else:
        clipped = list(g)
    new_velocity = [momentum * v + d for v, d in zip(velocity, clipped)]
    return [w - lr * v for w, v in zip(weights, new_velocity)], new_velocity
