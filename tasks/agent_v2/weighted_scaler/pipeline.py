from core import fit, transform

def solve(p):
    state = fit(p['other'], [1.0]*len(p['other']))
    if not state and p['other']:
        state = [[0.0, 1.0] for _ in p['other'][0]]
    return transform(p['other'], state)
