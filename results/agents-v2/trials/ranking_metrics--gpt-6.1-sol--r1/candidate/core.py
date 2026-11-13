from itertools import groupby


def validate(y, s, w):
    if (len(y) != len(s) or len(y) != len(w)
            or any(v not in (0, 1) for v in y)
            or any(v < 0 for v in w)):
        raise ValueError('inputs')


def auc(y, s, w):
    validate(y, s, w)
    pos = sum(v for a, v in zip(y, w) if a == 1)
    neg = sum(v for a, v in zip(y, w) if a == 0)
    if not pos or not neg:
        return None
    below = answer = 0.0
    rows = sorted(zip(y, s, w), key=lambda row: row[1])
    for _, group in groupby(rows, key=lambda row: row[1]):
        group_pos = group_neg = 0.0
        for label, score, weight in group:
            if label == 1:
                group_pos += weight
            else:
                group_neg += weight
        answer += (group_pos / pos) * ((below + 0.5 * group_neg) / neg)
        below += group_neg
    return answer


def ap(y, s, w):
    validate(y, s, w)
    total = sum(v for a, v in zip(y, w) if a == 1)
    if not total:
        return None
    tp = fp = answer = 0.0
    rows = sorted(zip(y, s, w), key=lambda row: row[1], reverse=True)
    for _, group in groupby(rows, key=lambda row: row[1]):
        delta = group_neg = 0.0
        for label, score, weight in group:
            if label == 1:
                delta += weight
            else:
                group_neg += weight
        tp += delta
        fp += group_neg
        if tp + fp:
            answer += (delta / total) * (tp / (tp + fp))
    return answer
