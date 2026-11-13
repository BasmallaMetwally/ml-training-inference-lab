from core import count, merge


def solve(p):
    state = count([], [], p['labels'])
    for shard in p['shards']:
        state = merge(state, count(shard['actual'], shard['predicted'], []))

    labels, matrix = state
    scores = []
    for i in range(len(labels)):
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        scores.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    macro_f1 = sum(scores) / len(scores) if scores else 0.0
    return {'labels': labels, 'matrix': matrix, 'macro_f1': macro_f1}
