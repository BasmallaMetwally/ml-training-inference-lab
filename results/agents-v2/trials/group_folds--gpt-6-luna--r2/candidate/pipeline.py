from core import assign

def solve(p):
    groups=p['groups']; k=p['k']
    mapping=assign(groups,k,0)
    return [[i for i,g in enumerate(groups) if mapping[g]==f] for f in range(k)]
