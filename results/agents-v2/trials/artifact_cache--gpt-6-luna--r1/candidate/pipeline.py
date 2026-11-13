from core import fingerprint, valid


def solve(p):
    config = p['config']
    return {
        'key': fingerprint(config),
        'reusable': valid(p['entry'], p['checkpoint'], p['graph'], config),
    }
