from core import windows


def solve(p):
    cutoff = p['cutoff']
    args = (p['lookback'], p['horizon'], p['stride'])
    train_rows = []
    eval_rows = []
    for row in p['rows']:
        if row['t'] < cutoff:
            train_rows.append(row)
        else:
            eval_rows.append(row)
    return {
        'train': windows(train_rows, *args),
        'eval': windows(eval_rows, *args),
    }
