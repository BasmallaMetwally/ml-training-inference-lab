from core import fit, transform


def solve(payload):
    train = payload['train']
    other = payload['other']
    state = fit(train, payload['weights'])
    if not train and other:
        width = len(other[0])
        state = [[0.0, 1.0] for _ in range(width)]
    return transform(other, state)
