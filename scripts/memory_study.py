"""Lightweight launcher: no Torch/NumPy imports before spawning memory workers."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run three exec-separated observations for every runtime and validation mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--out", default="results/performance/memory-corrected.json")
    args = parser.parse_args()
    if args.repetitions < 2:
        raise ValueError("At least two memory repetitions required")
    rows = []
    for repeat in range(args.repetitions):
        for name in ["eager", "fused", "compiled", "int8", "onnx"]:
            for checked in [True, False]:
                command = [
                    sys.executable,
                    "scripts/performance_study.py",
                    "--memory-worker",
                    name,
                    "--checked",
                    str(int(checked)),
                ]
                p = subprocess.run(
                    command, capture_output=True, text=True, timeout=240, check=False
                )
                lines = [
                    line.removeprefix("MEMORY_JSON=")
                    for line in p.stdout.splitlines()
                    if line.startswith("MEMORY_JSON=")
                ]
                if p.returncode == 0 and lines:
                    row = json.loads(lines[-1])
                    row["repetition"] = repeat
                    row["status"] = "measured"
                else:
                    row = {
                        "variant": name,
                        "checked": checked,
                        "repetition": repeat,
                        "status": "failed",
                        "error": p.stderr[-2000:].replace(str(Path.cwd()), "<project>"),
                    }
                rows.append(row)
                target = Path(args.out)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(
                        {
                            "status": "corrected_method",
                            "repetitions": args.repetitions,
                            "launcher": "stdlib-only process; workers exec before measurement",
                            "linux_counter": "/proc/self/status VmHWM; ru_maxrss diagnostic only",
                            "measurements": rows,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    name,
                    checked,
                    repeat,
                    row["status"],
                    row.get("peak_process_rss_bytes"),
                    flush=True,
                )


if __name__ == "__main__":
    main()
