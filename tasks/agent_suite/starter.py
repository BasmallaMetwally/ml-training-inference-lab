"""Repair all 12 functions to satisfy CONTRACTS.md. Do not change signatures."""
import torch


def stable_loss(logits, targets):
    return -(logits.exp() / logits.exp().sum(-1, keepdim=True))[torch.arange(len(targets)), targets].log().mean()


def masked_softmax(logits, mask):
    return logits.softmax(-1) * mask


def standardize(train, other):
    return (train-train.mean(0))/train.std(0), (other-other.mean(0))/other.std(0)


def pairwise_distance(x, y):
    return torch.cdist(x,y)


def ridge_fit(x, y, alpha):
    return torch.linalg.solve(x.T@x, x.T@y)


def clip_gradient(grads, max_norm):
    return grads * max_norm / grads.norm()


def accumulate_step(w, batches_x, batches_y, lr):
    gradient = sum(2*x.T@(x@w-y)/len(x) for x,y in zip(batches_x,batches_y))/len(batches_x)
    return w-lr*gradient


def predict(model, x):
    return model(x)


def adam_update(w, grad, m, v, step, lr, beta1, beta2, eps):
    m=beta1*m+(1-beta1)*grad
    v=beta2*v+(1-beta2)*grad.square()
    return w-lr*m/(v.sqrt()+eps), m, v


def confusion_matrix(targets, predictions, nclasses):
    return torch.bincount(predictions*nclasses+targets,minlength=nclasses*nclasses).reshape(nclasses,nclasses)


def stratified_split(labels, val_fraction, seed):
    indices=torch.randperm(len(labels))
    n=int(len(labels)*val_fraction)
    return indices[n:],indices[:n]


def early_stop(values, patience, min_delta):
    return min(range(len(values)),key=lambda i:values[i]),len(values)-1
