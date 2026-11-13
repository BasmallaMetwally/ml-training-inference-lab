from core import gradient,step

def solve(p):
    if p['clip']<0:
        raise ValueError('clip')
    w=list(p['weights']); v=list(p['velocity'])
    for batches in p['steps']:
        total=sum(len(b['x']) for b in batches)
        if total:
            g=[0.0]*len(w)
            for batch in batches:
                part=gradient(w,batch['x'],batch['y'])
                g=[a+b/len(batches) for a,b in zip(g,part)]
            w,v=step(w,v,g,p['lr'],p['momentum'],p['clip'])
    return {'weights':w,'velocity':v}
