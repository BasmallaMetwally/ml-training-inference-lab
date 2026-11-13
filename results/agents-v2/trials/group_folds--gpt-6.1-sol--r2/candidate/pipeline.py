from core import assign


def solve(p):
    groups = p['groups']
    k = p['k']
    mapping = assign(groups, k, p['seed'])
    folds = [[] for _ in range(k)]
    for index, group in enumerate(groups):
        folds[mapping[group]].append(index)
    return folds
