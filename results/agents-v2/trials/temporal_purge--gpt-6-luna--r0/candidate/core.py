def chronological(rows):
    """Return rows ordered by timestamp and then by string ID."""
    return sorted(rows, key=lambda row: (int(row['t']), str(row['id'])))


def split(rows, cutoff, gap):
    """Split complete entities around a temporal cutoff, purging the gap."""
    try:
        if not gap >= 0:
            raise ValueError('gap must be nonnegative')
    except (TypeError, ValueError) as exc:
        raise ValueError('gap must be nonnegative') from exc

    groups = {}
    for row in rows:
        groups.setdefault(row['entity'], []).append(int(row['t']))

    labels = {}
    for entity, timestamps in groups.items():
        if max(timestamps) < cutoff - gap:
            labels[entity] = 'train'
        elif min(timestamps) >= cutoff + gap:
            labels[entity] = 'eval'
        else:
            labels[entity] = 'purged'

    result = {'train': [], 'eval': [], 'purged': []}
    for row in chronological(rows):
        result[labels[row['entity']]].append(row['id'])
    return result
