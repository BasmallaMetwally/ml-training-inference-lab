from core import assign


def solve(p):
    groups = p['groups']
    k = p['k']
    mapping = assign(groups, k, p['seed'])
    return [[i for i, group in enumerate(groups) if mapping[group] == fold] for fold in range(k)]
