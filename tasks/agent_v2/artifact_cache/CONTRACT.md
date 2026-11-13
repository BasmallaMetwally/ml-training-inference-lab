# artifact_cache

core.fingerprint(config) is SHA256 of UTF-8 canonical JSON with sorted dictionary keys, separators comma/colon, ensure_ascii=False, allow_nan=False. core.valid(entry,checkpoint,graph,config) checks entry checkpoint_sha256 == SHA256 UTF8 checkpoint, graph_sha256 == SHA256 UTF8 graph and config_sha256 == fingerprint(config). Missing fields mean invalid. pipeline.solve returns {key:fingerprint(config), reusable:valid(...)}; stale graph/config must invalidate even if checkpoint matches. JSON types only, float NaN invalid. Preserve inputs.

Only core.py and pipeline.py may be modified. Tests, contracts and fixtures are protected. A passing visible test is not sufficient for acceptance.
