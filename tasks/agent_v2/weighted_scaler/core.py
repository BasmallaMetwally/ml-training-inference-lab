import math

def fit(rows, weights):
    if len(rows) != len(weights) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('weights')
    width = len(rows[0]) if rows else 0
    state = []
    for j in range(width):
        pairs = [(row[j], w) for row, w in zip(rows, weights) if row[j] is not None]
        total = sum(w for _, w in pairs)
        mean = sum(x*w for x,w in pairs)/total if total else 0.0
        var = sum(w*(x-mean)**2 for x,w in pairs)/max(total-1, 1) if total else 0.0
        state.append([mean, math.sqrt(var) if var > 0 else 1.0])
    return state

def transform(rows, state):
    return [[None if x is None else (x-state[j][0])/state[j][1] for j,x in enumerate(row)] for row in rows]
