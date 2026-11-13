import hashlib
import json


def fingerprint(config):
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def valid(entry, checkpoint, graph, config):
    if not isinstance(entry, dict):
        return False
    try:
        expected_checkpoint = hashlib.sha256(checkpoint.encode('utf-8')).hexdigest()
        expected_graph = hashlib.sha256(graph.encode('utf-8')).hexdigest()
        expected_config = fingerprint(config)
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        entry.get('checkpoint_sha256') == expected_checkpoint
        and entry.get('graph_sha256') == expected_graph
        and entry.get('config_sha256') == expected_config
    )
