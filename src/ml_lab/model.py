import copy

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_linear_bn_eval


class FashionMLP(nn.Module):
    """Small fully connected classifier with checked public input boundaries."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Validate a batch before computing class logits."""
        self.validate(x)
        return self.layers(x)

    @staticmethod
    def validate(x: torch.Tensor) -> None:
        """Reject empty, incorrectly shaped, integral or non-finite inputs."""
        if x.ndim != 2 or x.shape[1] != 784 or x.shape[0] == 0:
            raise ValueError("Expected a nonempty [batch, 784] tensor")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Inputs must be finite floating-point values")

    def forward_unchecked(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward for validation ablations; callers must validate inputs."""
        return self.layers(x)


class FashionCNN(FashionMLP):
    """Compact convolutional comparison, accepting the same flat image contract."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Unflatten(1, (1, 28, 28)),
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 10),
        )


def make_model(kind: str = "mlp") -> nn.Module:
    """Construct an architecture from its stable experiment identifier."""
    if kind == "mlp":
        return FashionMLP()
    if kind == "cnn":
        return FashionCNN()
    raise ValueError(f"Unknown model kind: {kind}")


def fused_model(model: FashionMLP) -> FashionMLP:
    """Fold running BatchNorm statistics into Linear weights for CPU inference."""
    if model.training:
        raise ValueError("Fusion requires eval mode")
    result = copy.deepcopy(model)
    for linear, bn in [(0, 1), (4, 5)]:
        result.layers[linear] = fuse_linear_bn_eval(result.layers[linear], result.layers[bn])
        result.layers[bn] = nn.Identity()
    return result.eval()
