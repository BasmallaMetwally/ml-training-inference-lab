"""Candidate-side tensor RPC worker. Contains no reference answers or graders."""

import contextlib
import importlib.util
import io
import json
import sys

import torch
from torch import nn


def decode(value):
    """Decode explicit CPU tensor envelopes and nested argument lists."""
    if isinstance(value, dict) and "tensor" in value:
        return torch.tensor(
            value["tensor"],
            dtype=getattr(torch, value["dtype"]),
            requires_grad=value.get("requires_grad", False),
        )
    if isinstance(value, list):
        return [decode(x) for x in value]
    return value


def encode(value):
    """Encode tensors without sharing Python objects with the parent evaluator."""
    if isinstance(value, torch.Tensor):
        return {
            "tensor": value.detach().tolist(),
            "dtype": str(value.dtype).removeprefix("torch."),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, (list, tuple)):
        return [encode(x) for x in value]
    return value


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("submission", "/submission/solution.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    requests = json.load(sys.stdin)
    responses = []
    for request in requests:
        try:
            torch.manual_seed(request["seed"])
            args = decode(request["args"])
            before = [x.detach().clone() if isinstance(x, torch.Tensor) else None for x in args]
            extra = {}
            if request["task"] == "predict":
                model = nn.Sequential(
                    nn.Linear(7, 16),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Dropout(0.8),
                    nn.Linear(16, 3),
                )
                state = model[1].running_mean.clone()
                args = [model, *args]
            with contextlib.redirect_stdout(io.StringIO()):
                output = getattr(module, request["task"])(*args)
            if request["task"] == "stable_loss":
                extra["gradient"] = encode(torch.autograd.grad(output, args[0])[0])
            if request["task"] == "predict":
                extra["batchnorm_unchanged"] = bool(torch.equal(model[1].running_mean, state))
            else:
                extra["inputs_unchanged"] = all(
                    v is None or torch.equal(v, args[i]) for i, v in enumerate(before)
                )
            responses.append({"status": "returned", "output": encode(output), **extra})
        except Exception as exc:
            responses.append({"status": "crash", "error": type(exc).__name__})
    print("RESULT_JSON=" + json.dumps(responses, allow_nan=True))


if __name__ == "__main__":
    main()
