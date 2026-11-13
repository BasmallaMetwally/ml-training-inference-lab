import math
import torch


def stable_loss(logits, targets):
    return torch.nn.functional.cross_entropy(logits, targets, reduction="mean")


def masked_softmax(logits, mask):
    mask = mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    row_has_valid = mask.any(dim=-1, keepdim=True)
    # Avoid softmax([-inf, ...]) for fully masked rows.
    safe_logits = torch.where(row_has_valid, masked_logits, torch.zeros_like(masked_logits))
    probabilities = torch.softmax(safe_logits, dim=-1)
    return torch.where(mask & row_has_valid, probabilities, torch.zeros_like(probabilities))


def standardize(train, other):
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (train - mean) / std, (other - mean) / std


def pairwise_distance(x, y):
    # The direct difference formulation avoids cancellation for repeated rows.
    distances = (x[:, None, :] - y[None, :, :]).square().sum(dim=-1)
    return distances.clamp_min(0)


def ridge_fit(x, y, alpha):
    n_features = x.shape[1]
    identity = torch.eye(n_features, dtype=x.dtype, device=x.device)
    return torch.linalg.solve(x.T @ x + alpha * identity, x.T @ y)


def clip_gradient(grads, max_norm):
    norm = torch.linalg.vector_norm(grads)
    scale = torch.clamp(torch.as_tensor(max_norm, dtype=grads.dtype, device=grads.device) / norm.clamp_min(torch.finfo(grads.dtype).tiny), max=1.0)
    return grads * scale


def accumulate_step(w, batches_x, batches_y, lr):
    total_count = sum(x.shape[0] for x in batches_x)
    if total_count == 0:
        return w.clone()
    grad = torch.zeros_like(w)
    for x, y in zip(batches_x, batches_y):
        residual = x @ w - y
        grad = grad + (2.0 / total_count) * (x.T @ residual)
    return w - lr * grad


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


def adam_update(w, grad, m, v, step, lr, beta1, beta2, eps):
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * grad.square()
    corrected_m = new_m / (1 - beta1 ** step)
    corrected_v = new_v / (1 - beta2 ** step)
    new_w = w - lr * corrected_m / (corrected_v.sqrt() + eps)
    return new_w, new_m, new_v


def confusion_matrix(targets, predictions, nclasses):
    result = torch.zeros((nclasses, nclasses), dtype=torch.int64, device=targets.device)
    if targets.numel():
        flat = targets * nclasses + predictions
        counts = torch.bincount(flat, minlength=nclasses * nclasses)
        result = counts.reshape(nclasses, nclasses)
    return result


def stratified_split(labels, val_fraction, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_parts = []
    val_parts = []
    for cls in torch.unique(labels, sorted=True):
        class_indices = torch.nonzero(labels == cls, as_tuple=False).flatten()
        perm = torch.randperm(class_indices.numel(), generator=generator)
        n_val = math.floor(class_indices.numel() * val_fraction)
        val_parts.append(class_indices[perm[:n_val]])
        train_parts.append(class_indices[perm[n_val:]])
    train = torch.cat(train_parts) if train_parts else torch.empty(0, dtype=torch.int64)
    val = torch.cat(val_parts) if val_parts else torch.empty(0, dtype=torch.int64)
    return train, val


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
