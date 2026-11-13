def count(actual, predicted, labels):
    if len(actual) != len(predicted) or len(labels) != len(set(labels)):
        raise ValueError('inputs')

    names = sorted(set(labels) | set(actual) | set(predicted))
    indices = {label: index for index, label in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for actual_label, predicted_label in zip(actual, predicted):
        matrix[indices[actual_label]][indices[predicted_label]] += 1
    return names, matrix


def merge(left, right):
    left_labels, left_matrix = left
    right_labels, right_matrix = right
    names = sorted(set(left_labels) | set(right_labels))
    indices = {label: index for index, label in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]

    for labels, values in ((left_labels, left_matrix), (right_labels, right_matrix)):
        for row, actual_label in enumerate(labels):
            for column, predicted_label in enumerate(labels):
                matrix[indices[actual_label]][indices[predicted_label]] += values[row][column]
    return names, matrix
