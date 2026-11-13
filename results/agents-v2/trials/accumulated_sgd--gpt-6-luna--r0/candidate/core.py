import math

def gradient(weights, x, y):
    if not x:
        return [0.0] * len(weights)
    return [
        2.0 * sum((sum(a * b for a, b in zip(row, weights)) - target) * row[j]
                  for row, target in zip(x, y)) / len(x)
        for j in range(len(weights))
    ]

def step(weights, velocity, g, lr, momentum, clip):
    if clip < 0:
        raise ValueError('clip')
    norm = math.sqrt(sum(value * value for value in g))
    if norm == 0 or clip == 0:
        scale = 0.0 if clip == 0 else 1.0
    else:
        scale = min(1.0, clip / norm)
    clipped = [value * scale for value in g]
    new_velocity = [momentum * old + grad for old, grad in zip(velocity, clipped)]
    new_weights = [weight - lr * vel for weight, vel in zip(weights, new_velocity)]
    return new_weights, new_velocity
