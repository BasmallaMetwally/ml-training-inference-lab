"""Collect actual model-generated repairs, with identical zero-feedback protocol."""

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from ml_lab.engine import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results/agents"
    out.mkdir(parents=True, exist_ok=True)
    scratch = root.parents[1] / "work/agent-workspaces"
    contracts = (root / "tasks/agent_suite/CONTRACTS.md").read_text()
    starter = (root / "tasks/agent_suite/starter.py").read_text()
    prompt = (
        "This is a controlled ML repair evaluation. Use no tools, shell commands, file reads or network. "
        "Return only a JSON object with one string key source containing a complete executable Python module. "
        "Repair all twelve functions below. Do not run tests or inspect graders. "
        "You may import torch, math and standard library modules.\n\n"
        + contracts
        + "\n\nSTARTER:\n"
        + starter
    )
    schema = {
        "type": "object",
        "properties": {"source": {"type": "string"}},
        "required": ["source"],
        "additionalProperties": False,
    }
    schema_path = out / "response_schema.json"
    write_json(schema_path, schema)
    (out / "prompt.txt").write_text(prompt)
    protocol = {
        "models": ["gpt-6-luna", "gpt-6.1-sol"],
        "repetitions": 3,
        "tasks": 12,
        "timeout_seconds_per_suite": 240,
        "feedback": "none",
        "tools": "forbidden; transcripts audited",
        "unit": "one suite response per model/repetition; 12 task outcomes share a response",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "reasoning_effort": "medium",
        "temperature": "provider default; not user-controlled through this CLI",
    }
    write_json(out / "protocol.json", protocol)
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("Codex CLI unavailable")
    for repetition in range(3):
        for model in protocol["models"]:
            trial = out / f"{model}-r{repetition}"
            if (trial / "metadata.json").exists():
                continue
            trial.mkdir(parents=True, exist_ok=True)
            workspace = scratch / f"{model}-r{repetition}"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "CONTRACTS.md").write_text(contracts)
            (workspace / "starter.py").write_text(starter)
            response = trial / "response.json"
            command = [
                codex,
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                model,
                "-c",
                'model_reasoning_effort="medium"',
                "--json",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(response),
                "--cd",
                str(workspace),
                "-",
            ]
            start = time.perf_counter()
            status = "generated"
            error = None
            with (
                (trial / "transcript.jsonl").open("w") as stdout,
                (trial / "stderr.log").open("w") as stderr,
            ):
                try:
                    process = subprocess.run(
                        command, input=prompt, text=True, stdout=stdout, stderr=stderr, timeout=240
                    )
                    if process.returncode:
                        status = "provider_error"
                except subprocess.TimeoutExpired:
                    status = "timeout"
            if response.exists():
                try:
                    source = json.loads(response.read_text())["source"]
                    compile(source, "solution.py", "exec")
                    (trial / "solution.py").write_text(source)
                except Exception as exc:
                    status = "invalid_response"
                    error = str(exc)
            elif status == "generated":
                status = "invalid_response"
            events = []
            for line in (trial / "transcript.jsonl").read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
            tool_events = [
                e
                for e in events
                if e.get("item", {}).get("type")
                in ["command_execution", "mcp_tool_call", "web_search", "file_change"]
            ]
            if tool_events:
                status = "protocol_violation"
            value = {
                "model": model,
                "repetition": repetition,
                "status": status,
                "seconds": time.perf_counter() - start,
                "tool_events": len(tool_events),
                "error": error,
                "prompt_sha256": protocol["prompt_sha256"],
                "cli_version": subprocess.run(
                    [codex, "--version"], capture_output=True, text=True
                ).stdout.strip(),
            }
            if (trial / "solution.py").exists():
                value["source_sha256"] = hashlib.sha256(
                    (trial / "solution.py").read_bytes()
                ).hexdigest()
            write_json(trial / "metadata.json", value)
            print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
