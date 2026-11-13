"""Numerically stable implementations of the twelve ML repair functions."""
import math
import torch


def stable_loss(logits, targets):
    return torch.nn.functional.cross_entropy(logits, targets, reduction="mean")


def masked_softmax(logits, mask):
    masked = logits.masked_fill(~mask, float("-inf"))
    row_max = masked.amax(dim=-1, keepdim=True)
    row_max = torch.where(mask.any(dim=-1, keepdim=True), row_max, torch.zeros_like(row_max))
    weights = (masked - row_max).exp()
    denominator = weights.sum(dim=-1, keepdim=True)
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return weights / denominator


def standardize(train, other):
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-6)
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
    scaled_norm = torch.linalg.vector_norm((grads / magnitude).to(torch.float64))
    factor = (float(max_norm) / magnitude.to(torch.float64) / scaled_norm).clamp(max=1.0)
    return grads * factor.to(dtype=grads.dtype)


def accumulate_step(w, batches_x, batches_y, lr):
    gradient_sum = torch.zeros_like(w)
    total = 0
    for x, y in zip(batches_x, batches_y):
        gradient_sum = gradient_sum + 2 * (x.T @ (x @ w - y))
        total += x.shape[0]
    if total == 0:
        return w.clone()
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
    generator.manual_seed(int(seed))
    train_parts = []
    validation_parts = []
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=True)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        count = math.floor(indices.numel() * val_fraction)
        validation_parts.append(indices[:count])
        train_parts.append(indices[count:])
    if not train_parts:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    return torch.cat(train_parts), torch.cat(validation_parts)


def early_stop(values, patience, min_delta):
    best_loss = float("inf")
    best_index = 0
    stale = 0
    stop_index = len(values) - 1
    for index, value in enumerate(values):
        loss = float(value)
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
