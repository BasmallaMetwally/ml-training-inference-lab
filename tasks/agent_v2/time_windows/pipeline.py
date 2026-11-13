from core import windows

def solve(p):
    rows=p['rows']; c=p['cutoff']
    args=(p['lookback'],p['horizon'],p['stride'])
    # Validate duplicates before partition too.
    windows(rows,*args)
    return {'train':windows([r for r in rows if r['t']<c],*args),'eval':[w for w in windows(rows,*args) if w['start']>=c]}
