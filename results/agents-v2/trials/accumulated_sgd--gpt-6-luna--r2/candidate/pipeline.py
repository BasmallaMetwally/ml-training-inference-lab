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
            sample_count = len(batch['x'])
            if sample_count == 0:
                continue
            batch_gradient = gradient(weights, batch['x'], batch['y'])
            for index, value in enumerate(batch_gradient):
                accumulated[index] += value * sample_count
        mean_gradient = [value / total_samples for value in accumulated]
        weights, velocity = step(
            weights, velocity, mean_gradient, p['lr'], p['momentum'], p['clip']
        )
    return {'weights': weights, 'velocity': velocity}
