from core import bins, brier


def solve(p):
    r, y, n = p['probabilities'], p['y'], p['n']
    w = p.get('weights', [1.0] * len(y))
    grouped = bins(r, y, w, n)
    total = sum(w)
    ece = (sum(b['weight'] * abs(b['accuracy'] - b['confidence'])
               for b in grouped) / total if total else 0.0)
    return {'bins': grouped, 'ece': ece, 'brier': brier(r, y, w)}
