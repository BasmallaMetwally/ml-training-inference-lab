"""Numerically stable implementations of the twelve ML repair functions."""
import math
import torch


def stable_loss(logits, targets):
    return torch.nn.functional.cross_entropy(logits, targets, reduction="mean")


def masked_softmax(logits, mask):
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    has_entries = mask.any(dim=-1, keepdim=True)
    safe_logits = torch.where(has_entries, masked_logits, torch.zeros_like(masked_logits))
    return torch.softmax(safe_logits, dim=-1).masked_fill(~mask, 0.0)


def standardize(train, other):
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (train - mean) / std, (other - mean) / std


def pairwise_distance(x, y):
    differences = x[:, None, :] - y[None, :, :]
    return differences.square().sum(dim=-1).clamp_min(0.0)


def ridge_fit(x, y, alpha):
    gram = x.T @ x
    identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(gram + alpha * identity, x.T @ y)


def clip_gradient(grads, max_norm):
    if grads.numel() == 0:
        return grads.clone()
    work = grads.to(torch.float64)
    largest = work.abs().max()
    if largest.item() == 0.0:
        return torch.zeros_like(grads)
    scaled = work / largest
    scaled_norm = torch.linalg.vector_norm(scaled)
    if largest <= max_norm / scaled_norm:
        return grads.clone()
    return (scaled * (max_norm / scaled_norm)).to(dtype=grads.dtype)


def accumulate_step(w, batches_x, batches_y, lr):
    gradient_sum = torch.zeros_like(w)
    count = 0
    for x, y in zip(batches_x, batches_y):
        gradient_sum = gradient_sum + 2.0 * (x.T @ (x @ w - y))
        count += x.shape[0]
    if count == 0:
        return w.clone()
    return w - lr * (gradient_sum / count)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).detach()


def adam_update(w, grad, m, v, step, lr, beta1, beta2, eps):
    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * grad.square()
    corrected_m = new_m / (1.0 - beta1 ** step)
    corrected_v = new_v / (1.0 - beta2 ** step)
    new_w = w - lr * corrected_m / (corrected_v.sqrt() + eps)
    return new_w, new_m, new_v


def confusion_matrix(targets, predictions, nclasses):
    indices = targets.reshape(-1) * nclasses + predictions.reshape(-1)
    return torch.bincount(indices, minlength=nclasses * nclasses).reshape(nclasses, nclasses)


def stratified_split(labels, val_fraction, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    train_parts = []
    validation_parts = []
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=True)[0]
        permutation = torch.randperm(indices.numel(), generator=generator)
        shuffled = indices[permutation]
        validation_count = math.floor(indices.numel() * val_fraction)
        validation_parts.append(shuffled[:validation_count])
        train_parts.append(shuffled[validation_count:])
    if not train_parts:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty.clone()
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
