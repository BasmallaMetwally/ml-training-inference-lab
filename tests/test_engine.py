import copy

import pytest
import torch
from torch import nn

from ml_lab.data import split_indices
from ml_lab.engine import evaluate, restore_checkpoint, save_checkpoint, seed_everything, train_step
from ml_lab.model import FashionMLP, fused_model


def test_stratified_split_is_disjoint_complete_and_repeatable():
    labels = torch.arange(10).repeat_interleave(60)
    train, val = split_indices(labels, 42)
    assert len(train) == 500 and len(val) == 100
    assert set(train.tolist()).isdisjoint(val.tolist())
    assert sorted(torch.cat((train, val)).tolist()) == list(range(600))
    assert torch.equal(train, split_indices(labels, 42)[0])
    assert not torch.equal(train, split_indices(labels, 43)[0])
    assert torch.equal(labels[val].bincount(), torch.full((10,), 10))


def test_training_learns_a_synthetic_decision_boundary():
    seed_everything(3)
    x = torch.randn(64, 784)
    y = (x[:, 0] > 0).long()
    model = nn.Linear(784, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial = evaluate(model, x, y)["loss"]
    for _ in range(20):
        train_step(model, optimizer, x, y)
    final = evaluate(model, x, y)
    assert final["loss"] < initial * 0.2
    assert final["accuracy"] > 0.95


@pytest.mark.parametrize(
    "bad",
    [
        torch.full((2, 784), float("nan")),
        torch.zeros(2, 783),
        torch.zeros(0, 784),
        torch.ones(2, 784, dtype=torch.int64),
    ],
)
def test_model_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        FashionMLP()(bad)


def test_nonfinite_gradients_fail_before_optimizer_step():
    seed_everything(4)
    model = nn.Linear(2, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = copy.deepcopy(model.state_dict())
    handle = model.weight.register_hook(lambda g: g * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        train_step(model, optimizer, torch.ones(4, 2), torch.zeros(4, dtype=torch.long))
    handle.remove()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_checkpoint_restores_exact_cpu_training_trajectory(tmp_path):
    seed_everything(11)
    model = FashionMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    g = torch.Generator().manual_seed(11)
    x, y = torch.randn(48, 784), torch.randint(0, 10, (48,))

    def epoch(m, opt, generator):
        for idx in torch.randperm(len(x), generator=generator).split(16):
            train_step(m, opt, x[idx], y[idx])

    epoch(model, optimizer, g)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, g, 1, 0.5, {}, {}, [])
    epoch(model, optimizer, g)
    expected = copy.deepcopy(model.state_dict())
    resumed = FashionMLP()
    opt = torch.optim.Adam(resumed.parameters(), lr=0.001)
    rg = torch.Generator()
    restore_checkpoint(path, resumed, opt, rg)
    epoch(resumed, opt, rg)
    for name, actual in resumed.state_dict().items():
        assert torch.equal(actual, expected[name]), name
    model.eval()
    resumed.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(x), resumed(x), rtol=0, atol=0)


def test_fusion_preserves_nontrivial_running_statistics():
    seed_everything(5)
    model = FashionMLP()
    for _ in range(5):
        model(torch.randn(32, 784) * 2 + 3)
    model.eval()
    fused = fused_model(model)
    with torch.inference_mode():
        for batch in [1, 17, 64]:
            x = torch.randn(batch, 784) * 2 + 3
            torch.testing.assert_close(model(x), fused(x), rtol=1e-4, atol=1e-5)


def test_fusion_rejects_training_mode():
    with pytest.raises(ValueError):
        fused_model(FashionMLP())


def test_evaluation_disables_dropout_and_keeps_batchnorm_statistics():
    seed_everything(2)
    model = FashionMLP()
    x = torch.randn(19, 784)
    y = torch.randint(0, 10, (19,))
    before = model.layers[1].running_mean.clone()
    first = evaluate(model, x, y)
    second = evaluate(model, x, y)
    assert first == second
    assert torch.equal(before, model.layers[1].running_mean)
    assert sum(map(sum, first["confusion_matrix"])) == len(y)
