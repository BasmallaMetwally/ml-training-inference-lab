import torch

from ml_lab.engine import train_step
from ml_lab.model import make_model


def test_cnn_supports_the_flat_image_contract_and_backpropagation():
    model = make_model("cnn")
    x = torch.randn(7, 784)
    y = torch.randint(0, 10, (7,))
    optimizer = torch.optim.Adam(model.parameters())
    original = model.layers[1].weight.detach().clone()
    train_step(model, optimizer, x, y)
    assert model(x).shape == (7, 10)
    assert not torch.equal(model.layers[1].weight, original)


def test_cnn_and_mlp_use_different_capacity():
    mlp = make_model("mlp")
    cnn = make_model("cnn")
    assert sum(p.numel() for p in cnn.parameters()) < sum(p.numel() for p in mlp.parameters())
