import hashlib
from collections import Counter


def assign(groups, k, seed):
    if k < 2:
        raise ValueError('k')
    counts = Counter(groups)
    order = sorted(
        counts,
        key=lambda group: (
            -counts[group],
            hashlib.sha256((str(seed) + ':' + group).encode()).hexdigest(),
            group,
        ),
    )
    sizes = [0] * k
    mapping = {}
    for group in order:
        fold = min(range(k), key=lambda index: (sizes[index], index))
        mapping[group] = fold
        sizes[fold] += counts[group]
    return mapping
