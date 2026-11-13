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
    return (
        entry.get('checkpoint_sha256')
        == hashlib.sha256(checkpoint.encode('utf-8')).hexdigest()
        and entry.get('graph_sha256')
        == hashlib.sha256(graph.encode('utf-8')).hexdigest()
        and entry.get('config_sha256') == fingerprint(config)
    )
