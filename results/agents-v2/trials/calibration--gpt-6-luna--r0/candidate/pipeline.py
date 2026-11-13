from core import bins, brier

def solve(p):
    rows = p['probabilities']
    labels = p['y']
    weights = p.get('weights', [1.0] * len(labels))
    count = p['n']
    grouped = bins(rows, labels, weights, count)
    total = sum(weights)
    ece = (sum(record['weight'] * abs(record['accuracy'] - record['confidence'])
               for record in grouped) / total) if total else 0.0
    return {'bins': grouped, 'ece': ece, 'brier': brier(rows, labels, weights)}
