"""Process RSS counters with Linux exec-local high-water provenance."""

import resource
import sys
from pathlib import Path


def parse_proc_status(text: str) -> dict[str, int]:
    """Read Linux mm RSS counters, converting the documented kB units to bytes."""
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in {"VmRSS", "VmHWM"}:
            number, unit = rest.split()
            if unit != "kB":
                raise ValueError(f"Unexpected /proc unit: {unit}")
            values[key] = int(number) * 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        raise ValueError("Missing Linux RSS/high-water fields")
    return values


def process_memory() -> dict:
    """Measure this process only; Linux ru_maxrss is diagnostic, never authoritative."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    diagnostic = int(raw if sys.platform == "darwin" else raw * 1024)
    if sys.platform == "linux":
        counters = parse_proc_status(Path("/proc/self/status").read_text())
        return {
            "rss_bytes": counters["VmRSS"],
            "peak_rss_bytes": counters["VmHWM"],
            "method": "/proc/self/status VmHWM",
            "diagnostic_ru_maxrss_bytes": diagnostic,
        }
    if sys.platform == "darwin":
        import psutil

        return {
            "rss_bytes": psutil.Process().memory_info().rss,
            "peak_rss_bytes": diagnostic,
            "method": "macOS RUSAGE_SELF ru_maxrss",
            "diagnostic_ru_maxrss_bytes": diagnostic,
        }
    raise RuntimeError(f"Unsupported memory-counter platform: {sys.platform}")
