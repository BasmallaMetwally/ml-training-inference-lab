import hashlib
import json

def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=False,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def valid(entry,checkpoint,graph,config):
    return (entry.get('checkpoint_sha256')==hashlib.sha256(checkpoint.encode()).hexdigest() and entry.get('config_sha256')==fingerprint(config))
