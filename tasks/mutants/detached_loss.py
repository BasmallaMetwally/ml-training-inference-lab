from torch.nn import functional as F


def stable_loss(logits, targets):
    return F.cross_entropy(logits, targets).detach()
