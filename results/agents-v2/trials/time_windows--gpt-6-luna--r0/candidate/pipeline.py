from core import windows


def solve(p):
    rows = p['rows']
    cutoff = p['cutoff']
    args = (p['lookback'], p['horizon'], p['stride'])

    # Partition raw records first so evaluation windows cannot use training history.
    train_rows = [row for row in rows if row['t'] < cutoff]
    eval_rows = [row for row in rows if row['t'] >= cutoff]
    return {
        'train': windows(train_rows, *args),
        'eval': windows(eval_rows, *args),
    }
