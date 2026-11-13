"""Repair these functions while preserving their signatures."""

import torch
from torch.nn import functional as F


def train_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    # BUG: training appears to run but parameters never update.
    return loss.item()


def stable_loss(logits, targets):
    # BUG: exp overflows for large finite logits.
    probabilities = logits.exp() / logits.exp().sum(1, keepdim=True)
    return -probabilities[torch.arange(len(targets)), targets].log().mean()


def predict(model, x):
    # BUG: dropout and BatchNorm behave as if training; graphs are retained.
    return model(x)
