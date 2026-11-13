def validate(y, s, w):
    if len(y) != len(s) or len(y) != len(w):
        raise ValueError('inputs')
    if any(v not in (0, 1) for v in y):
        raise ValueError('inputs')
    if any(v < 0 for v in w):
        raise ValueError('inputs')


def auc(y, s, w):
    validate(y, s, w)
    pos = sum(weight for label, weight in zip(y, w) if label == 1)
    neg = sum(weight for label, weight in zip(y, w) if label == 0)
    if pos == 0 or neg == 0:
        return None
    concordant = 0.0
    for yi, si, wi in zip(y, s, w):
        if yi != 1 or wi == 0:
            continue
        for yj, sj, wj in zip(y, s, w):
            if yj != 0 or wj == 0:
                continue
            if si > sj:
                concordant += wi * wj
            elif si == sj:
                concordant += 0.5 * wi * wj
    return concordant / (pos * neg)


def ap(y, s, w):
    validate(y, s, w)
    total = sum(weight for label, weight in zip(y, w) if label == 1)
    if total == 0:
        return None
    tp = fp = answer = 0.0
    for score in sorted(set(s), reverse=True):
        group = [(label, weight) for label, value, weight in zip(y, s, w) if value == score]
        delta = sum(weight for label, weight in group if label == 1)
        group_fp = sum(weight for label, weight in group if label == 0)
        tp += delta
        fp += group_fp
        if delta:
            answer += (delta / total) * (tp / (tp + fp))
    return answer
