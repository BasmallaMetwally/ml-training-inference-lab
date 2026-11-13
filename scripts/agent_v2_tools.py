"""Path-restricted file tools and isolated visible tests for the v2 experiment."""

import hashlib
import json
import math
import os
import resource
import selectors
import subprocess
import time
from pathlib import Path

EDITABLE = {"core.py", "pipeline.py"}
WORKER = Path(__file__).with_name("agent_v2_worker.py").resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    """Structural comparison; reject booleans masquerading as numeric answers."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (float, int)):
        return (
            type(actual) in (float, int)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-9)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(close(a, b) for a, b in zip(actual, expected))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(close(actual[k], v) for k, v in expected.items())
        )
    return False


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def bounded_process(command, payload, env, preexec):
    """Drain both pipes with a hard combined cap; never buffer unlimited output."""
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        preexec_fn=preexec,
    )
    streams = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        proc.stdin.write(payload.encode())
        proc.stdin.close()
        streams.register(proc.stdout, selectors.EVENT_READ, "stdout")
        streams.register(proc.stderr, selectors.EVENT_READ, "stderr")
        deadline = time.monotonic() + 12
        while streams.get_map():
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(command, 12)
            for key, _ in streams.select(timeout=0.1):
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    streams.unregister(key.fileobj)
                    continue
                buffers[key.data].extend(block)
                if sum(len(b) for b in buffers.values()) > 2**20:
                    raise ValueError("output exceeds 1 MiB")
        code = proc.wait(timeout=2)
        return subprocess.CompletedProcess(
            command, code, buffers["stdout"].decode(), buffers["stderr"].decode(errors="replace")
        )
    finally:
        streams.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=3)
        proc.stdout.close()
        proc.stderr.close()


def execute(workspace, payloads, backend="macos", image="ml-agent-v2:1"):
    """No expected values enter the candidate process or container."""
    workspace = Path(workspace).resolve()
    if backend == "docker":
        import uuid

        name = "ml-v2-" + uuid.uuid4().hex[:16]
        command = [
            "docker",
            "run",
            "--name",
            name,
            "--rm",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "32",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=8m",
            "--mount",
            f"type=bind,source={workspace},target=/submission,readonly",
            image,
        ]
    elif backend == "macos":
        # All user directories and temporary data denied except this exact trial.
        clauses = []
        for location in ["/Users", "/private/tmp", "/private/var/folders", "/private/etc"]:
            clauses.append(
                f'(deny file-read* (require-all (subpath "{location}") (require-not (subpath {json.dumps(str(workspace))}))))'
            )
        profile = (
            "(version 1)(allow default)(deny network*)(deny process-fork)(deny file-write*)"
            + "".join(clauses)
        )
        command = [
            "/usr/bin/sandbox-exec",
            "-p",
            profile,
            "/Library/Developer/CommandLineTools/usr/bin/python3",
            "-I",
            "-B",
            str(workspace / "_worker.py"),
            str(workspace),
        ]
    else:
        raise ValueError("An isolation backend is mandatory")
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    try:
        proc = bounded_process(
            command, json.dumps(payloads), env, limits if backend == "macos" else None
        )
        if proc.returncode:
            return {"status": "crash", "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
        if len(proc.stdout) > 2**20:
            return {"status": "invalid_output"}
        values = json.loads(proc.stdout)
        if not isinstance(values, list) or len(values) != len(payloads):
            return {"status": "invalid_output"}
        if any(not isinstance(v, dict) or type(v.get("mutated")) is not bool for v in values):
            return {"status": "invalid_output"}
        return {"status": "executed", "outputs": values}
    except subprocess.TimeoutExpired:
        if backend == "docker":
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
        return {"status": "timeout"}
    except (ValueError, OSError) as exc:
        return {"status": "invalid_output", "error": str(exc)[:300]}
    finally:
        if backend == "docker":
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)


def compare(execution, expected):
    """Require process success, exact structure, no mutation and numerical parity."""
    if execution["status"] != "executed":
        return {"status": execution["status"], "passed": 0, "total": len(expected)}
    passed = []
    for got, want in zip(execution["outputs"], expected):
        got = dict(got)
        mutated = got.pop("mutated")
        passed.append(not mutated and close(got, want))
    return {
        "status": "pass" if all(passed) else "incorrect",
        "passed": sum(passed),
        "total": len(expected),
        "case_passes": passed,
    }


class Tools:
    """Logical tools expose only a trial's files, never host shell or hidden grader."""

    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.protected = {
            p.name: digest(p)
            for p in self.workspace.iterdir()
            if p.is_file() and p.name not in EDITABLE
        }
        self.violations = []

    def call(self, action):
        kind, path = action["tool"], action["path"]
        if kind == "list_files":
            return {
                "files": sorted(p.name for p in self.workspace.iterdir() if p.name != "_worker.py")
            }
        if kind in ["read_file", "write_file"]:
            raw = self.workspace / path
            target = raw.resolve()
            if target.parent != self.workspace or raw.is_symlink() or path == "_worker.py":
                self.violations.append({"kind": "path_escape", "path": path})
                return {"error": "path denied"}
            if kind == "read_file":
                if not target.is_file():
                    return {"error": "file not found"}
                return {"content": target.read_text()[:30000]}
            if path not in EDITABLE:
                self.violations.append({"kind": "protected_write", "path": path})
                return {"error": "protected file; forbidden by task contract"}
            if len(action["content"]) > 40000:
                return {"error": "file size limit"}
            target.write_text(action["content"])
            return {"written": path, "sha256": digest(target)}
        if kind == "run_tests":
            fixtures = json.loads((self.workspace / "visible.json").read_text())
            execution = execute(self.workspace, [x["input"] for x in fixtures])
            result = compare(execution, [x["expected"] for x in fixtures])
            # Visible diagnostics only; final hidden outcomes are never fed back.
            result["observed"] = execution.get("outputs", [])
            result["expected"] = [x["expected"] for x in fixtures]
            return result
        if kind == "submit":
            return {"submitted": True}
        return {"error": "unknown tool"}

    def integrity(self):
        return all(
            (self.workspace / p).is_file() and digest(self.workspace / p) == h
            for p, h in self.protected.items()
        )
