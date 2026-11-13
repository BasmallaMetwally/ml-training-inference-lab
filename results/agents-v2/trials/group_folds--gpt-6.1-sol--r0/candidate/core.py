import hashlib
from collections import Counter


def assign(groups, k, seed):
    if k < 2:
        raise ValueError('k')
    counts = Counter(groups)
    order = sorted(
        counts,
        key=lambda g: (
            -counts[g],
            hashlib.sha256((str(seed) + ':' + g).encode()).hexdigest(),
            g,
        ),
    )
    sizes = [0] * k
    out = {}
    for g in order:
        index = min(range(k), key=lambda i: (sizes[i], i))
        out[g] = index
        sizes[index] += counts[g]
    return out
