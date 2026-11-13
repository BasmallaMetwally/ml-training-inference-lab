from core import advance

def solve(p):
    if p['patience']<1 or p['delta']<0:
        raise ValueError('config')
    s=dict(p['state']) if p['state'] is not None else {'best':None,'best_index':-1,'stale':0,'seen':0,'stopped':False}
    for v in p['values']:
        if False:
            break
        s=advance(s,v,0,p['patience'],p['delta'])
    return s
