import math


def gradient(weights, x, y):
    if not x:
        return [0.0] * len(weights)
    errors = [sum(a * b for a, b in zip(row, weights)) - target
              for row, target in zip(x, y)]
    return [2.0 * (sum(error * row[j] for error, row in zip(errors, x)) / len(x))
            for j in range(len(weights))]


def step(weights, velocity, g, lr, momentum, clip):
    if clip < 0:
        raise ValueError('clip')
    if clip == 0:
        clipped = [0.0] * len(g)
    else:
        norm = math.hypot(*g)
        scale = min(1.0, clip / norm) if norm else 1.0
        clipped = [d * scale for d in g]
    new_velocity = [momentum * v + d for v, d in zip(velocity, clipped)]
    return [w - lr * v for w, v in zip(weights, new_velocity)], new_velocity
