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
        checkpoint_sha256 = hashlib.sha256(checkpoint.encode('utf-8')).hexdigest()
        graph_sha256 = hashlib.sha256(graph.encode('utf-8')).hexdigest()
        config_sha256 = fingerprint(config)
    except (AttributeError, TypeError, ValueError, UnicodeEncodeError):
        return False
    return (
        entry.get('checkpoint_sha256') == checkpoint_sha256
        and entry.get('graph_sha256') == graph_sha256
        and entry.get('config_sha256') == config_sha256
    )
