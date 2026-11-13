from core import bins, brier


def solve(p):
    rows = p['probabilities']
    labels = p['y']
    weights = p.get('weights', [1.0] * len(labels))
    n = p['n']
    grouped = bins(rows, labels, weights, n)
    total = sum(weights)
    ece = (sum(bucket['weight'] * abs(bucket['accuracy'] - bucket['confidence'])
               for bucket in grouped) / total if total else 0.0)
    return {'bins': grouped, 'ece': ece, 'brier': brier(rows, labels, weights)}
