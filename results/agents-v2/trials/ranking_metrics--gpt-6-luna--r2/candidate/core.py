def validate(y, s, w):
    if len(y) != len(s) or len(y) != len(w):
        raise ValueError('inputs')
    if any(v not in (0, 1) for v in y) or any(v < 0 for v in w):
        raise ValueError('inputs')


def auc(y, s, w):
    validate(y, s, w)
    pos = sum(weight for label, weight in zip(y, w) if label == 1)
    neg = sum(weight for label, weight in zip(y, w) if label == 0)
    if pos == 0 or neg == 0:
        return None

    favorable = 0.0
    for label_i, score_i, weight_i in zip(y, s, w):
        if label_i != 1 or weight_i == 0:
            continue
        for label_j, score_j, weight_j in zip(y, s, w):
            if label_j == 0 and weight_j:
                if score_i > score_j:
                    favorable += weight_i * weight_j
                elif score_i == score_j:
                    favorable += 0.5 * weight_i * weight_j
    return favorable / (pos * neg)


def ap(y, s, w):
    validate(y, s, w)
    total_positive = sum(weight for label, weight in zip(y, w) if label == 1)
    if total_positive == 0:
        return None

    groups = {}
    for label, score, weight in zip(y, s, w):
        positive, negative = groups.get(score, (0, 0))
        if label == 1:
            positive += weight
        else:
            negative += weight
        groups[score] = (positive, negative)

    tp = fp = answer = 0.0
    for score in sorted(groups, reverse=True):
        positive, negative = groups[score]
        tp += positive
        fp += negative
        if positive and tp + fp:
            answer += (positive / total_positive) * (tp / (tp + fp))
    return answer
