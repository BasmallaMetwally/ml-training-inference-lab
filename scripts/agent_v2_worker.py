"""Untrusted-code worker. Expected outputs and grading logic stay in the host."""

import copy
import importlib
import json
import sys


def main():
    """Evaluate JSON inputs, report mutations and exceptions without an oracle."""
    requests = json.load(sys.stdin)
    sys.path.insert(0, "/submission" if len(sys.argv) < 2 else sys.argv[1])
    solve = importlib.import_module("pipeline").solve
    outputs = []
    for payload in requests:
        original = copy.deepcopy(payload)
        try:
            value = solve(payload)
            result = {"value": value}
        except Exception as exc:
            result = {"error": type(exc).__name__}
        result["mutated"] = payload != original
        outputs.append(result)
    print(json.dumps(outputs, allow_nan=False))


if __name__ == "__main__":
    main()
