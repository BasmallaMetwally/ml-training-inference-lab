"""Numerically stable implementations of the twelve ML repair functions."""
import math
import torch
import torch.nn.functional as F


def stable_loss(logits, targets):
    return F.cross_entropy(logits, targets, reduction="mean")


def masked_softmax(logits, mask):
    masked = logits.masked_fill(~mask, float("-inf"))
    has_entries = mask.any(dim=-1, keepdim=True)
    maximum = masked.amax(dim=-1, keepdim=True)
    maximum = torch.where(has_entries, maximum, torch.zeros_like(maximum))
    numerators = (masked - maximum).exp().masked_fill(~mask, 0)
    denominator = numerators.sum(dim=-1, keepdim=True)
    denominator = torch.where(has_entries, denominator, torch.ones_like(denominator))
    return numerators / denominator


def standardize(train, other):
    mean = train.mean(dim=0)
    std = train.std(dim=0, correction=0).clamp_min(1e-6)
    return (train - mean) / std, (other - mean) / std


def pairwise_distance(x, y):
    return (x[:, None, :] - y[None, :, :]).square().sum(dim=-1).clamp_min(0)


def ridge_fit(x, y, alpha):
    identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(x.T @ x + alpha * identity, x.T @ y)


def clip_gradient(grads, max_norm):
    if grads.numel() == 0:
        return grads.clone()
    magnitude = grads.abs().max()
    if magnitude.item() == 0:
        return torch.zeros_like(grads)
    scaled = grads / magnitude
    scaled_norm = torch.linalg.vector_norm(scaled)
    scale = ((max_norm / magnitude) / scaled_norm).clamp(max=1)
    return grads * scale


def accumulate_step(w, batches_x, batches_y, lr):
    total = 0
    gradient_sum = torch.zeros_like(w)
    for x, y in zip(batches_x, batches_y):
        gradient_sum = gradient_sum + 2 * (x.T @ (x @ w - y))
        total += x.shape[0]
    return w - lr * (gradient_sum / total)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).detach()


def adam_update(w, grad, m, v, step, lr, beta1, beta2, eps):
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * grad.square()
    corrected_m = new_m / (1 - beta1 ** step)
    corrected_v = new_v / (1 - beta2 ** step)
    new_w = w - lr * corrected_m / (corrected_v.sqrt() + eps)
    return new_w, new_m, new_v


def confusion_matrix(targets, predictions, nclasses):
    encoded = targets.reshape(-1) * nclasses + predictions.reshape(-1)
    return torch.bincount(encoded, minlength=nclasses * nclasses).reshape(nclasses, nclasses)


def stratified_split(labels, val_fraction, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_parts = []
    validation_parts = []
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=True)[0]
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        count = math.floor(indices.numel() * val_fraction)
        validation_parts.append(shuffled[:count])
        train_parts.append(shuffled[count:])
    if not train_parts:
        empty = torch.empty(0, dtype=torch.int64)
        return empty.clone(), empty.clone()
    return torch.cat(train_parts), torch.cat(validation_parts)


def early_stop(values, patience, min_delta):
    best_loss = float("inf")
    best_index = 0
    stale = 0
    stop_index = len(values) - 1
    for index, loss in enumerate(values):
        loss = float(loss)
        if loss < best_loss - min_delta:
            best_loss = loss
            best_index = index
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            stop_index = index
            break
    return best_index, stop_index
