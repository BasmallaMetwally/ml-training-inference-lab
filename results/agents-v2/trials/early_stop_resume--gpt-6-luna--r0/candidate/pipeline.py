from core import advance


def solve(p):
    patience = p['patience']
    delta = p['delta']
    if not patience >= 1 or not delta >= 0:
        raise ValueError('config')

    supplied = p['state']
    s = dict(supplied) if supplied is not None else {
        'best': None,
        'best_index': -1,
        'stale': 0,
        'seen': 0,
        'stopped': False,
    }
    if s['stopped']:
        return s

    values = iter(p['values'])
    while not s['stopped']:
        try:
            value = next(values)
        except StopIteration:
            break
        s = advance(s, value, s['seen'], patience, delta)
    return s
