def windows(rows, lookback, horizon, stride):
    if min(lookback, horizon, stride) < 1:
        raise ValueError('config')
    groups = {}
    seen = set()
    for row in rows:
        key = (row['entity'], row['t'])
        if key in seen:
            raise ValueError('duplicate')
        seen.add(key)
        groups.setdefault(row['entity'], []).append(row)

    out = []
    length = lookback + horizon
    for entity in sorted(groups):
        data = sorted(groups[entity], key=lambda row: row['t'])
        for offset in range(0, len(data) - length + 1, stride):
            part = data[offset:offset + length]
            if any(part[i]['t'] - part[i - 1]['t'] != 1
                   for i in range(1, length)):
                continue
            out.append({
                'entity': entity,
                'start': part[lookback]['t'],
                'x': [row['value'] for row in part[:lookback]],
                'y': [row['value'] for row in part[lookback:]],
            })
    return out
