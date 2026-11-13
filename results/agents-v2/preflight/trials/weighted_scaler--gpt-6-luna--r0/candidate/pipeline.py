from core import fit, transform


def solve(p):
    train = p['train']
    other = p['other']
    state = fit(train, p['weights'])
    if not train and other:
        state = [[0.0, 1.0] for _ in range(len(other[0]))]
    return transform(other, state)
