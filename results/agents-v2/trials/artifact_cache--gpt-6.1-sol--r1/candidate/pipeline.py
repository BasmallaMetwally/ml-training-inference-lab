from core import fingerprint, valid


def solve(p):
    return {
        'key': fingerprint(p['config']),
        'reusable': valid(p['entry'], p['checkpoint'], p['graph'], p['config']),
    }
