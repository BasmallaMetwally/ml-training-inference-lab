from core import fit, transform


def solve(p):
    train = p['train']
    weights = p['weights']
    other = p['other']
    # Validate weights even when the training set is empty.
    state = fit(train, weights)
    if not train:
        if not other:
            return []
        state = [[0.0, 1.0] for _ in other[0]]
    return transform(other, state)
