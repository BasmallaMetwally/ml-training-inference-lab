def count(actual, predicted, labels):
    if len(actual) != len(predicted) or len(labels) != len(set(labels)):
        raise ValueError('inputs')
    names = sorted(set(labels) | set(actual) | set(predicted))
    indices = {name: i for i, name in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for actual_label, predicted_label in zip(actual, predicted):
        matrix[indices[actual_label]][indices[predicted_label]] += 1
    return names, matrix


def merge(left, right):
    names = sorted(set(left[0]) | set(right[0]))
    indices = {name: i for i, name in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for labels, values in (left, right):
        positions = [indices[label] for label in labels]
        for i, actual_index in enumerate(positions):
            for j, predicted_index in enumerate(positions):
                matrix[actual_index][predicted_index] += values[i][j]
    return names, matrix
