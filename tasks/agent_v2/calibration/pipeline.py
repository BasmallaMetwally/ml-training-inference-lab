from core import bins,brier

def solve(p):
    r,y,w,n=p['probabilities'],p['y'],[1.0]*len(p['y']),p['n']
    grouped=bins(r,y,w,n); total=sum(w)
    ece=sum(b['weight']*abs(b['accuracy']-b['confidence']) for b in grouped)/total if total else 0.0
    return {'bins':grouped,'ece':ece,'brier':brier(r,y,w)}
