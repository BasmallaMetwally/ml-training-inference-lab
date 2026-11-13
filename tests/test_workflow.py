import pytest
import torch

from ml_lab import engine


def tiny_data(root, seed):
    g = torch.Generator().manual_seed(seed)
    # 129 exercises the singleton-batch merge at batch size 128.
    return (
        torch.randn(129, 784, generator=g),
        torch.randint(0, 10, (129,), generator=g),
        torch.randn(20, 784, generator=g),
        torch.randint(0, 10, (20,), generator=g),
        {"mean": 0.0, "std": 1.0, "seed": seed},
    )


def test_full_train_resume_matches_uninterrupted_run(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "training_data", tiny_data)
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resumed"
    engine.train(None, uninterrupted, epochs=2)
    engine.train(None, resumed, epochs=1)
    engine.train(None, resumed, epochs=2, resume=True)
    a = torch.load(uninterrupted / "last.pt", weights_only=True)
    b = torch.load(resumed / "last.pt", weights_only=True)
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    assert a["best_loss"] == b["best_loss"]


def test_resume_rejects_changed_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "training_data", tiny_data)
    engine.train(None, tmp_path, epochs=1)
    with pytest.raises(ValueError, match="configuration/data"):
        engine.train(None, tmp_path, epochs=2, lr=0.01, resume=True)


def test_existing_run_cannot_be_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "training_data", tiny_data)
    engine.train(None, tmp_path, epochs=1)
    with pytest.raises(ValueError, match="already exists"):
        engine.train(None, tmp_path, epochs=2)
