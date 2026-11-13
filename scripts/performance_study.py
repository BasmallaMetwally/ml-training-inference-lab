"""Ablations, alternative runtimes, operator profiling and real process RSS peaks."""

import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".cache/inductor").resolve()))
if sys.platform == "darwin":
    os.environ.setdefault("CXX", "/usr/bin/clang++")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import numpy as np
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.benchmark import Timer

from ml_lab.artifacts import export_cache_valid
from ml_lab.data import digest, training_data
from ml_lab.engine import environment, load_model, seed_everything, write_json
from ml_lab.model import FashionMLP, fused_model

OUT = Path("results/performance")


class Guarded(nn.Module):
    """Keep input validation identical across Torch and ONNX forward paths."""

    def __init__(self, core: Callable, checked: bool) -> None:
        super().__init__()
        self.core = core
        self.checked = checked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same input boundary before invoking the selected runtime."""
        if self.checked:
            FashionMLP.validate(x)
        return self.core(x)


def make_core(name: str) -> Callable:
    """Build an eval-only runtime from the fixed, selected seed42 checkpoint."""
    model, _ = load_model("runs/seed42")
    if name == "eager":
        return model.layers
    if name == "fused":
        return fused_model(model).layers
    if name == "compiled":
        return torch.compile(fused_model(model).layers, fullgraph=True)
    if name == "int8":
        engines = torch.backends.quantized.supported_engines
        torch.backends.quantized.engine = "x86" if "x86" in engines else "qnnpack"
        # Dynamic quantization is deliberately evaluated, not assumed equivalent.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return torch.ao.quantization.quantize_dynamic(
                fused_model(model).layers, {nn.Linear}, dtype=torch.qint8
            )
    if name == "onnx":
        import onnxruntime as ort

        path = OUT / "model.onnx"
        checkpoint_sha256 = digest("runs/seed42/best.pt")
        if not export_cache_valid(path, checkpoint_sha256):
            torch.onnx.export(
                fused_model(model).layers,
                (torch.zeros(2, 784),),
                str(path),
                input_names=["pixels"],
                output_names=["logits"],
                dynamic_shapes=({0: torch.export.Dim("batch")},),
                opset_version=18,
                external_data=False,
            )
            write_json(
                path.with_suffix(".onnx.json"),
                {"checkpoint_sha256": checkpoint_sha256, "graph_sha256": digest(path)},
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

        def forward(x):
            return torch.from_numpy(session.run(None, {"pixels": x.numpy()})[0])

        return forward
    raise ValueError(name)


def memory_worker(name: str, checked: bool) -> None:
    """Fresh-process OS high-water RSS including imports/model/runtime, not just tensors."""
    import psutil

    seed_everything(42)
    baseline = psutil.Process().memory_info().rss
    core = make_core(name)
    model = Guarded(core, checked)
    sample = torch.randn(128, 784)
    with torch.inference_mode():
        for _ in range(50):
            model(sample)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    value = {
        "variant": name,
        "checked": checked,
        "batch_size": 128,
        "peak_process_rss_bytes": peak_bytes,
        "baseline_rss_after_imports_bytes": baseline,
        "final_process_rss_bytes": psutil.Process().memory_info().rss,
        "scope": "Fresh process OS ru_maxrss: imports + model/runtime construction + 50 forwards. Compiled variant includes compilation; not activation-only memory.",
    }
    print("MEMORY_JSON=" + json.dumps(value))


def main() -> None:
    """Measure runtimes, validation ablations, operator attribution and RSS."""
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    model, ck = load_model("runs/seed42")
    _, _, x, y, _ = training_data("data/fashion-mnist", 42)
    with torch.inference_mode():
        reference = torch.cat([model(v) for v in x.split(512)])
    results = {
        "environment": environment(),
        "checkpoint_sha256": digest("runs/seed42/best.pt"),
        "protocol": {
            "repeats": 7,
            "min_run_time": 0.15,
            "warmup": 20,
            "order_seed": 2026,
            "parity_policy": "zero validation prediction disagreements required for accepted speedups",
            "latency_scope": "CPU in-memory forward including NumPy bridge for ONNX; excludes runtime setup/compilation",
        },
        "variants": {},
    }
    jobs = []
    for name in ["eager", "fused", "compiled", "int8", "onnx"]:
        before = time.perf_counter()
        try:
            core = make_core(name)
            with torch.inference_mode():
                # Warm each measured batch shape before validation and timing.
                for batch in [1, 32, 128, 512]:
                    core(x[:batch])
                actual = torch.cat([core(v) for v in x.split(512)])
            error = float((actual - reference).abs().max())
            disagree = int((actual.argmax(1) != reference.argmax(1)).sum())
            value = {
                "status": "measured",
                "setup_and_warmup_seconds": time.perf_counter() - before,
                "max_abs_logit_error": error,
                "prediction_disagreements": disagree,
                "validation_accuracy": float((actual.argmax(1) == y).float().mean()),
                "accepted_exact_predictions": disagree == 0,
                "measurements": [],
            }
            for checked in [True, False]:
                wrapper = Guarded(core, checked)
                for batch in [1, 32, 128, 512]:
                    jobs.append((name, checked, batch, wrapper))
            results["variants"][name] = value
        except Exception as exc:
            results["variants"][name] = {
                "status": "unsupported_or_failed",
                "error": f"{type(exc).__name__}: {exc}"[-3000:]
                .replace(str(Path.cwd()), "<project>")
                .replace(sys.prefix, "<venv>")
                .replace(str(Path.home()), "<home>"),
            }
        write_json(OUT / "summary.json", results)
    samples = {(name, checked, b): [] for name, checked, b, _ in jobs}
    rng = random.Random(2026)
    for repeat in range(7):
        rng.shuffle(jobs)
        for name, checked, batch, wrapper in jobs:
            sample = x[:batch].contiguous()
            with torch.inference_mode():
                for _ in range(20):
                    wrapper(sample)
                measurement = Timer(
                    "model(x)", globals={"model": wrapper, "x": sample}, num_threads=1
                ).blocked_autorange(min_run_time=0.15)
            samples[name, checked, batch].append(
                {
                    "repeat": repeat,
                    "median_ms": measurement.median * 1000,
                    "iqr_ms": measurement.iqr * 1000,
                    "number_per_run": measurement.number_per_run,
                    "raw_block_seconds": measurement.raw_times,
                }
            )
    for (name, checked, batch), trials in samples.items():
        results["variants"][name]["measurements"].append(
            {
                "checked": checked,
                "batch_size": batch,
                "median_ms": float(np.median([r["median_ms"] for r in trials])),
                "trials": trials,
            }
        )
    write_json(OUT / "summary.json", results)
    # Profiler attribution is instrumented and is not substituted for benchmark latency.
    profiles = {}
    for name in ["eager", "fused"]:
        core = make_core(name)
        for checked in [True, False]:
            key = f"{name}-" + ("checked" if checked else "unchecked")
            sample = x[:1]
            with torch.inference_mode():
                for _ in range(20):
                    Guarded(core, checked)(sample)
                with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                    for _ in range(100):
                        with record_function("whole_forward"):
                            if checked:
                                with record_function("input_validation"):
                                    FashionMLP.validate(sample)
                            with record_function("network_core"):
                                core(sample)
            prof.export_chrome_trace(str(OUT / f"{key}-trace.json"))
            rows = [
                {
                    "name": e.key,
                    "count": e.count,
                    "self_cpu_us": e.self_cpu_time_total,
                    "total_cpu_us": e.cpu_time_total,
                }
                for e in prof.key_averages()
            ]
            profiles[key] = {
                "forwards": 100,
                "operators": rows,
                "aten_calls": sum(e["count"] for e in rows if e["name"].startswith("aten::")),
                "note": "Self CPU time includes operator/dispatch implementation overhead; this profiler does not separately identify pure Python vs C++ dispatch cost.",
            }
    write_json(OUT / "profiles.json", profiles)
    memory = []
    for name, value in results["variants"].items():
        if value["status"] != "measured":
            continue
        for checked in [True, False]:
            p = subprocess.run(
                [sys.executable, __file__, "--memory-worker", name, "--checked", str(int(checked))],
                capture_output=True,
                text=True,
                timeout=180,
            )
            lines = [
                line.removeprefix("MEMORY_JSON=")
                for line in p.stdout.splitlines()
                if line.startswith("MEMORY_JSON=")
            ]
            memory.append(
                json.loads(lines[-1])
                if lines
                else {
                    "variant": name,
                    "checked": checked,
                    "status": "failed",
                    "error": p.stderr[-2000:],
                }
            )
    write_json(
        OUT / "memory.json",
        {
            "measurements": memory,
            "method": "OS peak RSS in separate processes; platform units normalized to bytes",
        },
    )
    print(
        "Performance ablation, runtime parity, profiler traces and peak RSS complete.", flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--memory-worker")
    p.add_argument("--checked", type=int, default=1)
    args = p.parse_args()
    if args.memory_worker:
        memory_worker(args.memory_worker, bool(args.checked))
    else:
        main()
