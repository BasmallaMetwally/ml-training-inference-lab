from torch.nn import functional as F


def train_step(model, optimizer, x, y):
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()
