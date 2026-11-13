def count(actual, predicted, labels):
    if len(actual) != len(predicted) or len(labels) != len(set(labels)):
        raise ValueError('inputs')

    names = sorted(set(labels) | set(actual) | set(predicted))
    indices = {label: i for i, label in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for actual_label, predicted_label in zip(actual, predicted):
        matrix[indices[actual_label]][indices[predicted_label]] += 1
    return names, matrix


def merge(left, right):
    names = sorted(set(left[0]) | set(right[0]))
    indices = {label: i for i, label in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]

    for labels, values in (left, right):
        for i, actual_label in enumerate(labels):
            for j, predicted_label in enumerate(labels):
                matrix[indices[actual_label]][indices[predicted_label]] += values[i][j]
    return names, matrix
