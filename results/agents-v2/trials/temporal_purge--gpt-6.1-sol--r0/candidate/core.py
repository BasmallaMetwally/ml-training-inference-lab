def chronological(rows):
    return sorted(rows, key=lambda row: (int(row['t']), str(row['id'])))


def split(rows, cutoff, gap):
    try:
        valid_gap = gap >= 0
    except (TypeError, ValueError):
        raise ValueError('gap') from None
    if not valid_gap:
        raise ValueError('gap')

    rows = list(rows)
    bounds = {}
    for row in rows:
        entity = row['entity']
        timestamp = int(row['t'])
        if entity in bounds:
            low, high = bounds[entity]
            bounds[entity] = (min(low, timestamp), max(high, timestamp))
        else:
            bounds[entity] = (timestamp, timestamp)

    labels = {}
    for entity, (low, high) in bounds.items():
        if high < cutoff - gap:
            labels[entity] = 'train'
        elif low >= cutoff + gap:
            labels[entity] = 'eval'
        else:
            labels[entity] = 'purged'

    result = {'train': [], 'eval': [], 'purged': []}
    for row in chronological(rows):
        result[labels[row['entity']]].append(row['id'])
    return result
