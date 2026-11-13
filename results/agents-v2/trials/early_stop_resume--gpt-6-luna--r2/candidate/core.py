import math


def _validate_config(patience, delta):
    try:
        valid_patience = patience >= 1
        valid_delta = delta >= 0
    except (TypeError, ValueError, OverflowError):
        raise ValueError('config') from None
    if not valid_patience or not valid_delta:
        raise ValueError('config')


def advance(state, value, index, patience, delta):
    _validate_config(patience, delta)
    try:
        finite_value = math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('value') from None
    if not finite_value:
        raise ValueError('value')

    out = dict(state)
    if out['best'] is None or value < out['best'] - delta:
        out.update(best=value, best_index=index, stale=0)
    else:
        out['stale'] += 1
    out['seen'] += 1
    out['stopped'] = out['stale'] >= patience
    return out
