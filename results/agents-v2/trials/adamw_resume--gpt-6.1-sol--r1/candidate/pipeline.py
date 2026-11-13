from core import update


def solve(p):
    w, m, v = list(p['weights']), list(p['m']), list(p['v'])
    t = p['step']
    for g in p['gradients']:
        t += 1
        w, m, v = update(
            w, m, v, g, t, p['lr'], p['b1'], p['b2'], p['eps'], p['decay']
        )
    return {'weights': w, 'm': m, 'v': v, 'step': t}
