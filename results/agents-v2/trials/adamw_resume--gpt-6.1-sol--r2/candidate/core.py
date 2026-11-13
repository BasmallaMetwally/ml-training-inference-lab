import math


def update(w, m, v, g, t, lr, b1, b2, eps, decay):
    nm = [b1 * old_m + (1 - b1) * grad for old_m, grad in zip(m, g)]
    nv = [b2 * old_v + (1 - b2) * grad * grad for old_v, grad in zip(v, g)]
    correction_m = 1 - b1 ** t
    correction_v = 1 - b2 ** t
    nw = [
        old_w * (1 - lr * decay)
        - lr * (new_m / correction_m)
        / (math.sqrt(new_v / correction_v) + eps)
        for old_w, new_m, new_v in zip(w, nm, nv)
    ]
    return nw, nm, nv
