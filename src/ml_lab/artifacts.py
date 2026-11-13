"""Verify exported model caches against the source checkpoint and graph bytes."""

import json
from pathlib import Path

from .data import digest


def export_cache_valid(graph: str | Path, checkpoint_sha256: str) -> bool:
    """A cached graph is reusable only if its provenance and bytes both match."""
    graph = Path(graph)
    metadata = graph.with_suffix(graph.suffix + ".json")
    if not graph.exists() or not metadata.exists():
        return False
    try:
        value = json.loads(metadata.read_text())
        return value["checkpoint_sha256"] == checkpoint_sha256 and value["graph_sha256"] == digest(
            graph
        )
    except (OSError, ValueError, KeyError):
        return False
