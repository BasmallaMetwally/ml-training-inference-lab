def count(actual, predicted, labels):
    if len(actual) != len(predicted) or len(labels) != len(set(labels)):
        raise ValueError('inputs')
    names = sorted(set(labels) | set(actual) | set(predicted))
    indices = {name: i for i, name in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for a, b in zip(actual, predicted):
        matrix[indices[a]][indices[b]] += 1
    return names, matrix


def merge(left, right):
    names = sorted(set(left[0]) | set(right[0]))
    indices = {name: i for i, name in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for labels, values in (left, right):
        positions = [indices[label] for label in labels]
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                matrix[positions[i]][positions[j]] += value
    return names, matrix
