from core import windows

def solve(p):
    rows=p['rows']; c=p['cutoff']
    args=(p['lookback'],p['horizon'],p['stride'])
    # Validate duplicates before partition too.
    windows(rows,*args)
    train_rows=[r for r in rows if r['t']<c]
    eval_rows=[r for r in rows if r['t']>=c]
    return {'train':windows(train_rows,*args),'eval':windows(eval_rows,*args)}
