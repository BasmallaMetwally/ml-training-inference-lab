import json

from ml_lab.artifacts import export_cache_valid
from ml_lab.data import digest


def test_export_cache_rejects_a_different_checkpoint_and_modified_graph(tmp_path):
    graph = tmp_path / "model.onnx"
    graph.write_bytes(b"fixture graph bytes")
    metadata = graph.with_suffix(".onnx.json")
    metadata.write_text(
        json.dumps({"checkpoint_sha256": "checkpoint-A", "graph_sha256": digest(graph)})
    )
    assert export_cache_valid(graph, "checkpoint-A")
    assert not export_cache_valid(graph, "checkpoint-B")
    graph.write_bytes(b"different graph bytes")
    assert not export_cache_valid(graph, "checkpoint-A")


def test_export_cache_without_provenance_is_rejected(tmp_path):
    graph = tmp_path / "model.onnx"
    graph.write_bytes(b"untracked")
    assert not export_cache_valid(graph, "checkpoint-A")
