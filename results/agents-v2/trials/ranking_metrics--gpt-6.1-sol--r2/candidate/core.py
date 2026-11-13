def validate(y, s, w):
    if (len(y) != len(s) or len(y) != len(w)
            or any(v not in (0, 1) for v in y)
            or any(v < 0 for v in w)):
        raise ValueError('inputs')


def _groups(y, score, w):
    rows = sorted(zip(score, y, w), key=lambda row: row[0])
    i = 0
    while i < len(rows):
        threshold = rows[i][0]
        positive = negative = 0
        j = i
        while j < len(rows) and rows[j][0] == threshold:
            _, label, weight = rows[j]
            if label == 1:
                positive += weight
            else:
                negative += weight
            j += 1
        yield positive, negative
        i = j


def auc(y, score, w):
    validate(y, score, w)
    pos = sum(weight for label, weight in zip(y, w) if label == 1)
    neg = sum(weight for label, weight in zip(y, w) if label == 0)
    if pos == 0 or neg == 0:
        return None
    below = 0
    answer = 0.0
    for positive, negative in _groups(y, score, w):
        answer += (positive / pos) * ((below + negative / 2) / neg)
        below += negative
    return answer


def ap(y, score, w):
    validate(y, score, w)
    total = sum(weight for label, weight in zip(y, w) if label == 1)
    if total == 0:
        return None
    tp = fp = answer = 0.0
    for positive, negative in reversed(list(_groups(y, score, w))):
        tp += positive
        fp += negative
        if positive:
            answer += (positive / total) * (tp / (tp + fp))
    return answer
