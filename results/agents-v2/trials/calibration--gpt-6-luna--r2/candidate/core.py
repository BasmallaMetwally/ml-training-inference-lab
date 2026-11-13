import math

def validate(rows,y,w,n):
    if n<1 or len(rows)!=len(y) or len(y)!=len(w) or any(v<0 for v in w):
        raise ValueError('inputs')
    for r,t in zip(rows,y):
        if not r or not 0<=t<len(r) or any(not math.isfinite(v) or v<0 or v>1 for v in r) or abs(sum(r)-1)>1e-8:
            raise ValueError('probabilities')

def bins(rows,y,w,n):
    validate(rows,y,w,n)
    accum=[[0.,0.,0.] for _ in range(n)]
    for r,t,v in zip(rows,y,w):
        pred=max(range(len(r)),key=lambda i:r[i]); conf=r[pred]
        slot=int(conf*n)
        accum[slot][0]+=v; accum[slot][1]+=v*(pred==t); accum[slot][2]+=v*conf
    return [{'weight':a,'accuracy':b/a if a else 0.,'confidence':c/a if a else 0.} for a,b,c in accum]

def brier(rows,y,w):
    total=sum(w)
    return sum(v*sum((a-(j==t))**2 for j,a in enumerate(r)) for r,t,v in zip(rows,y,w))/total if total else 0.0
