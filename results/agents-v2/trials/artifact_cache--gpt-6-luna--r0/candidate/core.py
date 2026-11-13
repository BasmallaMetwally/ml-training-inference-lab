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

    expected = {
        'checkpoint_sha256': hashlib.sha256(checkpoint.encode('utf-8')).hexdigest(),
        'graph_sha256': hashlib.sha256(graph.encode('utf-8')).hexdigest(),
        'config_sha256': fingerprint(config),
    }
    return all(entry.get(field) == digest for field, digest in expected.items())
