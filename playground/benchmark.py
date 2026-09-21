#!/usr/bin/env python3
"""Run a reproducible local baseline benchmark for one model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from playground.image import launcher as image_launcher
from playground.llm import assets as llm_assets
from playground.llm import launcher as llm_launcher
from playground.paths import LOGS_ROOT


LLM_MODELS = set(llm_assets.profile_names())
IMAGE_MODELS = set(image_launcher.MODELS)
REPORTS = LOGS_ROOT / "benchmarks"
PROMPT = (
    "Explain in three concise bullet points why local inference benefits from reproducible "
    "benchmarks. Mention latency, memory, and output quality."
)


def report_path(model: str, destination: str | None = None) -> Path:
    if destination:
        return Path(destination).expanduser()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPORTS / f"{stamp}-{model}.json"


def write_report(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def post(port: int, body: dict, timeout: int = 900) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def resident_mib(pid: int) -> float | None:
    try:
        value = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True,
            timeout=5, check=False,
        ).stdout.strip()
        return round(int(value) / 1024, 1) if value else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def llm_benchmark(model_name: str, context: int | None, runs: int, destination: str | None) -> int:
    defaults = llm_launcher.profile_defaults(model_name)
    context = context or defaults["context"]
    if context < 4096:
        raise ValueError("--context must be at least 4096")
    llm_assets.install(model_name)
    _, model, adapter, _, server = llm_assets.paths(model_name)
    port = llm_launcher.free_port()
    command = llm_launcher.build_command(server, model, adapter, port, context, model_name)
    REPORTS.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS / f"{model_name}-server.log"
    result = {
        "schema": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "kind": "llm", "model": model_name, "context": context, "runs": [],
        "defaults": defaults, "command": [str(item) for item in command],
    }
    process = None
    old_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            isolation = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                         if os.name == "nt" else {"start_new_session": True})
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=server.parent, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, **isolation)
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            deadline = started + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Text backend exited with code {process.returncode}; see {log_path}")
                if llm_launcher.healthy(port):
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError(f"Text backend startup timed out; see {log_path}")
            result["startup_seconds"] = round(time.monotonic() - started, 2)
            post(port, {"messages": [{"role": "user", "content": "Reply only: ready"}],
                        "max_tokens": 8, "temperature": 0,
                        "chat_template_kwargs": {"enable_thinking": False}})
            for index in range(runs):
                started_run = time.monotonic()
                response = post(port, {
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": 128, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                choice = response.get("choices", [{}])[0]
                result["runs"].append({
                    "run": index + 1,
                    "seconds": round(time.monotonic() - started_run, 2),
                    "usage": response.get("usage"),
                    "timings": response.get("timings"),
                    "finish_reason": choice.get("finish_reason"),
                    "output_chars": len(choice.get("message", {}).get("content", "")),
                    "resident_mib": resident_mib(process.pid),
                })
            result["passed"] = all(item["output_chars"] > 0 for item in result["runs"])
    except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
        result["passed"] = False
        result["error"] = str(error)
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        llm_launcher.stop_backend(process)
    destination_path = report_path(model_name, destination)
    write_report(destination_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Benchmark report: {destination_path}")
    return 0 if result.get("passed") else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(LLM_MODELS | IMAGE_MODELS), default="compact")
    parser.add_argument("--runs", type=int, default=2, help="timed runs after one warm-up (LLM) or model load (image)")
    parser.add_argument("--context", type=int, help="LLM context; defaults to the selected profile")
    parser.add_argument("--steps", type=int, help="image steps; defaults to the selected profile")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--report", help="JSON report path")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.model in LLM_MODELS:
        try:
            return llm_benchmark(args.model, args.context, args.runs, args.report)
        except (OSError, RuntimeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
    forwarded = ["--model", args.model, "--no-prompt", "--no-browser", "--benchmark",
                 "--runs", str(args.runs), "--width", str(args.width), "--height", str(args.height),
                 "--seed", str(args.seed)]
    if args.steps is not None:
        forwarded += ["--steps", str(args.steps)]
    if args.report:
        forwarded += ["--report", args.report]
    return image_launcher.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
