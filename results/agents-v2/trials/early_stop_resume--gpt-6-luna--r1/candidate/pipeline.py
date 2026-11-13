from core import advance


def solve(p):
    patience = p['patience']
    delta = p['delta']
    if patience < 1 or not delta >= 0:
        raise ValueError('config')

    state = p['state']
    s = dict(state) if state is not None else {
        'best': None,
        'best_index': -1,
        'stale': 0,
        'seen': 0,
        'stopped': False,
    }
    for value in p['values']:
        if s['stopped']:
            break
        s = advance(s, value, s['seen'], patience, delta)
    return s
