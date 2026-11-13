from core import advance, _validate_config


def solve(p):
    _validate_config(p['patience'], p['delta'])
    supplied = p['state']
    s = dict(supplied) if supplied is not None else {
        'best': None,
        'best_index': -1,
        'stale': 0,
        'seen': 0,
        'stopped': False,
    }
    for v in p['values']:
        if s['stopped']:
            break
        s = advance(s, v, s['seen'], p['patience'], p['delta'])
    return s
