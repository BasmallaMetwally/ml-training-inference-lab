from core import gradient, step

def solve(p):
    if p['clip'] < 0:
        raise ValueError('clip')
    weights = list(p['weights'])
    velocity = list(p['velocity'])
    for batches in p['steps']:
        total_samples = sum(len(batch['x']) for batch in batches)
        if total_samples == 0:
            continue
        accumulated = [0.0] * len(weights)
        for batch in batches:
            count = len(batch['x'])
            if count == 0:
                continue
            part = gradient(weights, batch['x'], batch['y'])
            for j, value in enumerate(part):
                accumulated[j] += value * count / total_samples
        weights, velocity = step(weights, velocity, accumulated,
                                 p['lr'], p['momentum'], p['clip'])
    return {'weights': weights, 'velocity': velocity}
