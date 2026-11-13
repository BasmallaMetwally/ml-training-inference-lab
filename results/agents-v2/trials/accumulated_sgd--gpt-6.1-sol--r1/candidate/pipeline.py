from core import gradient, step


def solve(p):
    if p['clip'] < 0:
        raise ValueError('clip')
    w = list(p['weights'])
    v = list(p['velocity'])
    for batches in p['steps']:
        total = sum(len(batch['x']) for batch in batches)
        if not total:
            continue
        g = [0.0] * len(w)
        for batch in batches:
            count = len(batch['x'])
            if not count:
                continue
            part = gradient(w, batch['x'], batch['y'])
            fraction = count / total
            g = [a + b * fraction for a, b in zip(g, part)]
        w, v = step(w, v, g, p['lr'], p['momentum'], p['clip'])
    return {'weights': w, 'velocity': v}
