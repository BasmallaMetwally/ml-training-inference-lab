def chronological(rows):
    """Return rows ordered by timestamp, then by string ID."""
    return sorted(rows, key=lambda row: (int(row['t']), str(row['id'])))


def split(rows, cutoff, gap):
    """Assign each entity wholly to training, evaluation, or purged."""
    try:
        if gap < 0 or gap != gap:
            raise ValueError('gap must be nonnegative')
    except (TypeError, OverflowError):
        raise ValueError('gap must be nonnegative')

    groups = {}
    for row in rows:
        groups.setdefault(row['entity'], []).append(row['t'])

    labels = {}
    for entity, timestamps in groups.items():
        earliest = min(timestamps)
        latest = max(timestamps)
        if latest < cutoff - gap:
            labels[entity] = 'train'
        elif earliest >= cutoff + gap:
            labels[entity] = 'eval'
        else:
            labels[entity] = 'purged'

    result = {'train': [], 'eval': [], 'purged': []}
    for row in chronological(rows):
        result[labels[row['entity']]].append(row['id'])
    return result
