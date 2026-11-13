"""Repairs for a compact PyTorch ML engineering suite."""
import math
import torch


def stable_loss(logits, targets):
    return torch.nn.functional.cross_entropy(logits, targets, reduction="mean")


def masked_softmax(logits, mask):
    mask = mask.to(dtype=torch.bool)
    # Handle fully masked rows separately so their output is exactly zero.
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    any_valid = mask.any(dim=-1, keepdim=True)
    row_max = masked_logits.max(dim=-1, keepdim=True).values
    row_max = torch.where(any_valid, row_max, torch.zeros_like(row_max))
    exps = torch.exp(masked_logits - row_max).masked_fill(~mask, 0)
    denom = exps.sum(dim=-1, keepdim=True)
    return exps / denom.clamp_min(torch.finfo(exps.dtype).tiny)


def standardize(train, other):
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (train - mean) / std, (other - mean) / std


def pairwise_distance(x, y):
    # The direct difference formulation avoids cancellation for repeated or
    # nearly identical rows that can occur with the norm expansion.
    return (x[:, None, :] - y[None, :, :]).square().sum(dim=-1).clamp_min(0)


def ridge_fit(x, y, alpha):
    n_features = x.shape[1]
    system = x.T @ x + alpha * torch.eye(n_features, dtype=x.dtype, device=x.device)
    return torch.linalg.solve(system, x.T @ y)


def clip_gradient(grads, max_norm):
    norm = torch.linalg.vector_norm(grads)
    scale = torch.clamp(torch.as_tensor(max_norm, dtype=grads.dtype, device=grads.device) / norm.clamp_min(torch.finfo(grads.dtype).tiny), max=1.0)
    return grads * scale


def accumulate_step(w, batches_x, batches_y, lr):
    total = sum(x.shape[0] for x in batches_x)
    gradient = torch.zeros_like(w)
    for x, y in zip(batches_x, batches_y):
        gradient = gradient + 2 * x.T @ (x @ w - y)
    gradient = gradient / total
    return w - lr * gradient


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


def adam_update(w, grad, m, v, step, lr, beta1, beta2, eps):
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * grad.square()
    m_hat = new_m / (1 - beta1 ** step)
    v_hat = new_v / (1 - beta2 ** step)
    new_w = w - lr * m_hat / (v_hat.sqrt() + eps)
    return new_w, new_m, new_v


def confusion_matrix(targets, predictions, nclasses):
    encoded = targets * nclasses + predictions
    return torch.bincount(encoded, minlength=nclasses * nclasses).reshape(nclasses, nclasses).to(torch.int64)


def stratified_split(labels, val_fraction, seed):
    generator = torch.Generator(device=labels.device)
    generator.manual_seed(seed)
    train_parts = []
    val_parts = []
    for cls in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == cls, as_tuple=False).flatten()
        perm = torch.randperm(indices.numel(), generator=generator, device=labels.device)
        n_val = math.floor(indices.numel() * val_fraction)
        val_parts.append(indices[perm[:n_val]])
        train_parts.append(indices[perm[n_val:]])
    train = torch.cat(train_parts) if train_parts else torch.empty(0, dtype=torch.int64, device=labels.device)
    val = torch.cat(val_parts) if val_parts else torch.empty(0, dtype=torch.int64, device=labels.device)
    # Shuffle the final lists deterministically as well.
    train = train[torch.randperm(train.numel(), generator=generator, device=labels.device)]
    val = val[torch.randperm(val.numel(), generator=generator, device=labels.device)]
    return train.to(torch.int64), val.to(torch.int64)


def early_stop(values, patience, min_delta):
    best_loss = values[0]
    best_index = 0
    stale = 0
    stop_index = len(values) - 1
    for i in range(1, len(values)):
        if values[i] < best_loss - min_delta:
            best_loss = values[i]
            best_index = i
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                stop_index = i
                break
    return best_index, stop_index
