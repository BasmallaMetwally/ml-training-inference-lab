def chronological(rows):
    """Return rows ordered by timestamp, then identifier, without mutating input."""
    return sorted(rows, key=lambda row: (row['t'], row['id']))


def split(rows, cutoff, gap):
    """Split complete entities around a cutoff, purging groups in the gap."""
    try:
        if not gap >= 0:
            raise ValueError('gap must be nonnegative')
    except TypeError as exc:
        raise ValueError('gap must be nonnegative') from exc

    bounds = {}
    for row in rows:
        entity = row['entity']
        timestamp = row['t']
        if entity not in bounds:
            bounds[entity] = [timestamp, timestamp]
        else:
            bounds[entity][0] = min(bounds[entity][0], timestamp)
            bounds[entity][1] = max(bounds[entity][1], timestamp)

    labels = {}
    for entity, (minimum, maximum) in bounds.items():
        if maximum < cutoff - gap:
            labels[entity] = 'train'
        elif minimum >= cutoff + gap:
            labels[entity] = 'eval'
        else:
            labels[entity] = 'purged'

    result = {'train': [], 'eval': [], 'purged': []}
    for row in chronological(rows):
        result[labels[row['entity']]].append(row['id'])
    return result
