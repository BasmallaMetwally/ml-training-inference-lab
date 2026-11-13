def chronological(rows):
    return sorted(rows, key=lambda r: (r['t'], r['id']))

def split(rows, cutoff, gap):
    if gap < 0:
        raise ValueError('gap')
    groups = {}
    for row in rows:
        groups.setdefault(row['entity'], []).append(row['t'])
    labels = {g: ('train' if min(ts) <= cutoff-gap else 'eval' if max(ts) > cutoff+gap else 'purged') for g,ts in groups.items()}
    result = {k: [] for k in ['train','eval','purged']}
    for row in chronological(rows):
        result[labels[row['entity']]].append(row['id'])
    return result
