def count(actual, predicted, labels):
    if len(actual) != len(predicted):
        raise ValueError('actual and predicted lengths differ')
    if len(labels) != len(set(labels)):
        raise ValueError('duplicate initial labels')

    names = sorted(set(labels) | set(actual) | set(predicted))
    index = {label: i for i, label in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for actual_label, predicted_label in zip(actual, predicted):
        matrix[index[actual_label]][index[predicted_label]] += 1
    return names, matrix


def merge(left, right):
    left_labels, left_matrix = left
    right_labels, right_matrix = right
    names = sorted(set(left_labels) | set(right_labels))
    index = {label: i for i, label in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]

    for labels, values in ((left_labels, left_matrix), (right_labels, right_matrix)):
        for row, actual_label in enumerate(labels):
            for column, predicted_label in enumerate(labels):
                matrix[index[actual_label]][index[predicted_label]] += values[row][column]
    return names, matrix
