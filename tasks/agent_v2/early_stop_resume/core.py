import math

def advance(state,value,index,patience,delta):
    if not math.isfinite(value):
        raise ValueError('value')
    out=dict(state)
    if out['best'] is None or value<=out['best']-delta:
        out.update(best=value,best_index=index,stale=0)
    else:
        out['stale']+=1
    out['seen']+=1
    out['stopped']=out['stale']>=patience
    return out
