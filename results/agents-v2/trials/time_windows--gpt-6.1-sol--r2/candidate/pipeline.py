from core import windows


def solve(p):
    cutoff = p['cutoff']
    train_rows = []
    eval_rows = []
    for row in p['rows']:
        if row['t'] < cutoff:
            train_rows.append(row)
        else:
            eval_rows.append(row)
    args = (p['lookback'], p['horizon'], p['stride'])
    return {
        'train': windows(train_rows, *args),
        'eval': windows(eval_rows, *args),
    }
