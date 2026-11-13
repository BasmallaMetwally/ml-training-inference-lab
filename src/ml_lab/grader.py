"""Functional candidate checks in a timeout-limited local subprocess.

This is not a security sandbox. Run only trusted candidate code.
"""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path


def grade(task: str, candidate: str | Path, timeout: float = 30) -> dict:
    """Run public development fixtures in a trusted local subprocess."""
    if task not in {"train_step", "stable_loss", "predict"} or timeout <= 0:
        raise ValueError("Unknown task or invalid timeout")
    path = Path(candidate).resolve(strict=True)
    before = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ml_lab.grader", task, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"task": task, "status": "timeout", "seconds": time.perf_counter() - before}
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        result = {"task": task, "status": "crash", "detail": proc.stderr[-2000:]}
    if proc.returncode and result.get("status") == "pass":
        result = {"task": task, "status": "crash", "detail": proc.stderr[-2000:]}
    result["seconds"] = time.perf_counter() - before
    return result


def check(task: str, candidate: str | Path) -> dict:
    """Evaluate a legacy candidate against public seeded development cases."""
    import torch
    from torch import nn

    from .engine import seed_everything

    spec = importlib.util.spec_from_file_location("candidate", candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, task)
    for seed in [17, 53, 109]:
        seed_everything(seed)
        if task == "train_step":
            x = torch.randn(32, 7)
            y = (x[:, 0] > 0).long()
            model = nn.Linear(7, 2)
            opt = torch.optim.SGD(model.parameters(), lr=0.15)
            initial = nn.functional.cross_entropy(model(x), y).item()
            for _ in range(25):
                loss = function(model, opt, x, y)
                assert isinstance(loss, (float, int)) and torch.isfinite(torch.tensor(loss)), (
                    "Invalid loss"
                )
            final = nn.functional.cross_entropy(model(x), y).item()
            assert final < initial * 0.7, "Loss did not fall sufficiently"
            # Stale gradients must be cleared even if they already contain NaNs.
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, float("nan"))
            function(model, opt, x, y)
            assert all(torch.isfinite(p).all() for p in model.parameters()), (
                "Stale gradients were not cleared"
            )
            assert all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
            ), "Non-finite gradients"
        elif task == "stable_loss":
            for magnitude in [1.0, 1000.0, 100000.0]:
                logits = (torch.randn(19, 4) * magnitude).requires_grad_()
                target = torch.arange(19) % 4
                loss = function(logits, target)
                oracle = nn.functional.cross_entropy(logits, target)
                assert loss.ndim == 0 and torch.isfinite(loss), "Loss must be a finite scalar"
                torch.testing.assert_close(loss, oracle, atol=1e-5, rtol=1e-5)
                grad = torch.autograd.grad(loss, logits)[0]
                expected = torch.autograd.grad(oracle, logits)[0]
                torch.testing.assert_close(grad, expected, atol=1e-6, rtol=1e-5)
        else:
            model = nn.Sequential(
                nn.Linear(7, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.8), nn.Linear(16, 3)
            )
            x = torch.randn(23, 7)
            model.eval()
            with torch.inference_mode():
                expected = model(x).clone()
            model.train()
            before = model[1].running_mean.clone()
            first, second = function(model, x), function(model, x)
            assert not first.requires_grad and not second.requires_grad, (
                "Inference created a gradient graph"
            )
            torch.testing.assert_close(first, expected)
            torch.testing.assert_close(second, expected)
            torch.testing.assert_close(model[1].running_mean, before)
            assert first.shape == (23, 3), "Wrong prediction shape"
    return {"task": task, "status": "pass", "seed_cases": 3}


if __name__ == "__main__":
    task, candidate = sys.argv[1:]
    try:
        result = check(task, candidate)
    except (AssertionError, FloatingPointError) as exc:
        result = {"task": task, "status": "incorrect", "detail": str(exc)[-2000:]}
    except Exception as exc:
        result = {"task": task, "status": "crash", "detail": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
