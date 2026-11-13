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
    wins = 0.0
    for label_i, score_i, weight_i in zip(y, s, w):
        if label_i != 1 or weight_i == 0:
            continue
        for label_j, score_j, weight_j in zip(y, s, w):
            if label_j != 0 or weight_j == 0:
                continue
            if score_i > score_j:
                wins += weight_i * weight_j
            elif score_i == score_j:
                wins += 0.5 * weight_i * weight_j
    return wins / (pos * neg)


def ap(y, s, w):
    validate(y, s, w)
    total = sum(weight for label, weight in zip(y, w) if label == 1)
    if total == 0:
        return None
    tp = fp = answer = 0.0
    for score in sorted(set(s), reverse=True):
        group_positive = sum(weight for label, value, weight in zip(y, s, w) if value == score and label == 1)
        group_negative = sum(weight for label, value, weight in zip(y, s, w) if value == score and label == 0)
        tp += group_positive
        fp += group_negative
        if tp + fp:
            answer += (group_positive / total) * (tp / (tp + fp))
    return answer
