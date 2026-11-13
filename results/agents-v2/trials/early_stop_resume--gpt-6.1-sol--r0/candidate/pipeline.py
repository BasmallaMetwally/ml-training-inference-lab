from core import advance


def solve(p):
    patience = p['patience']
    delta = p['delta']
    if not patience >= 1 or not delta >= 0:
        raise ValueError('config')
    s = dict(p['state']) if p['state'] is not None else {
        'best': None, 'best_index': -1, 'stale': 0,
        'seen': 0, 'stopped': False,
    }
    if s['stopped']:
        return s
    for value in p['values']:
        s = advance(s, value, s['seen'], patience, delta)
        if s['stopped']:
            break
    return s
