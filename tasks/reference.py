"""Reference solutions. Keep outside the agent workspace during an evaluation."""

import torch
from torch.nn import functional as F


def train_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()


def stable_loss(logits, targets):
    return F.cross_entropy(logits, targets)


def predict(model, x):
    model.eval()
    with torch.inference_mode():
        return model(x)
