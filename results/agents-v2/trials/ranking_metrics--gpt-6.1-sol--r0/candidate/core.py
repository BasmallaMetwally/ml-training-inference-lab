from itertools import groupby


def validate(y, s, w):
    if (len(y) != len(s) or len(y) != len(w)
            or any(v not in (0, 1) for v in y)
            or any(v < 0 for v in w)):
        raise ValueError('inputs')


def _groups(y, s, w):
    rows = sorted(zip(s, y, w), key=lambda row: row[0], reverse=True)
    for _, group in groupby(rows, key=lambda row: row[0]):
        positive = negative = 0.0
        for score, label, weight in group:
            if label == 1:
                positive += weight
            else:
                negative += weight
        yield positive, negative


def auc(y, s, w):
    validate(y, s, w)
    pos = sum(weight for label, weight in zip(y, w) if label == 1)
    neg = sum(weight for label, weight in zip(y, w) if label == 0)
    if not pos or not neg:
        return None
    # Each negative pairs with all higher-scoring positives and half
    # of the positive weight at its own threshold.
    higher_positive = answer = 0.0
    for positive, negative in _groups(y, s, w):
        answer += (negative / neg) * ((higher_positive + positive / 2) / pos)
        higher_positive += positive
    return answer


def ap(y, s, w):
    validate(y, s, w)
    total = sum(weight for label, weight in zip(y, w) if label == 1)
    if not total:
        return None
    tp = fp = answer = 0.0
    for positive, negative in _groups(y, s, w):
        tp += positive
        fp += negative
        if positive:
            answer += (positive / total) * (tp / (tp + fp))
    return answer
