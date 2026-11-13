from core import count,merge

def solve(p):
    state=count([],[],p['labels'])
    for shard in p['shards']:
        state=merge(state,count(shard['actual'],shard['predicted'],[]))
    labels,m=state
    f=[]
    for i in range(len(labels)):
        denom=sum(m[i])+sum(row[i] for row in m)
        f.append(2*m[i][i]/denom if denom else 0.0)
    return {'labels':labels,'matrix':m,'macro_f1':sum(f)}
