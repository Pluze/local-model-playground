#!/usr/bin/env python3
"""Run isolated inference checks and an optional long-context capacity check."""
import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playground.llm import assets as install
from playground.llm import launcher as launch
from playground.paths import LOGS_ROOT, PROJECT_ROOT

ROOT = PROJECT_ROOT


def post(port, endpoint, body, timeout=900):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(port, prompt):
    return post(port, "/v1/chat/completions/input_tokens", {"messages": [{"role": "user", "content": prompt}]})["input_tokens"]


def target_prompt(port, target):
    phrase = "The local conversation should retain key facts and answer clearly. "
    low, high = 1, max(2, target // 4)
    while token_count(port, phrase * high) < target:
        high *= 2
    while low < high:
        middle = (low + high + 1) // 2
        if token_count(port, phrase * middle) <= target:
            low = middle
        else:
            high = middle - 1
    prompt = phrase * low
    return prompt, token_count(port, prompt)


def test_reasoning_control(port):
    body = {
        "messages": [{"role": "user", "content": "Calculate 17 times 19. Give only the result."}],
        "stream": True,
        "reasoning_control": True,
        "max_tokens": 128,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    result = {"id_received": False, "control_success": False, "final_content": False}
    sent = False
    with urllib.request.urlopen(request, timeout=120) as stream:
        for line in stream:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            item = json.loads(line[6:])
            completion_id = item.get("id")
            result["id_received"] |= bool(completion_id)
            delta = item.get("choices", [{}])[0].get("delta", {})
            if delta.get("reasoning_content") and not sent:
                if not completion_id:
                    raise RuntimeError("Reasoning stream lacks a completion ID")
                control = post(port, "/v1/chat/completions/control",
                               {"action": "reasoning_end", "id": completion_id}, timeout=30)
                result["control_success"] = control.get("success") is True
                result["control_response"] = control
                sent = True
            if delta.get("content"):
                result["final_content"] = True
    return result


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--profile", choices=install.profile_names(),
                        default=install.default_profile(), help="model profile to validate")
    parser.add_argument("--context", type=int, help="test a window at about 75%% occupancy (takes longer)")
    args = parser.parse_args(argv)
    if args.context is not None and args.context < 4096:
        parser.error("--context must be at least 4096")
    if launch.port_busy(8080) or launch.port_busy(8081):
        print("A local chat service is already running. Stop it before isolated validation.", file=sys.stderr)
        return 1
    if not install.check(args.model):
        print(f"Run python3 launch.py llm --model {args.model} first.", file=sys.stderr)
        return 1
    _, model, adapter, _, server = install.paths(args.model)
    context = args.context or launch.profile_defaults(args.model)["context"]
    port = launch.free_port()
    command = launch.build_command(server, model, adapter, port, context, args.model)
    logs = LOGS_ROOT / "llm"
    logs.mkdir(parents=True, exist_ok=True)
    result = {"platform": install.platform_key(), "profile": args.model, "context": context,
              "model": str(model), "adapter": str(adapter) if adapter else None, "runtime": str(server)}
    with (logs / "verify-server.log").open("w") as output:
        isolation = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                     if sys.platform == "win32" else {"start_new_session": True})
        process = subprocess.Popen(command, cwd=server.parent, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, **isolation)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, interrupt)
        try:
            started = time.monotonic()
            deadline = started + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {process.returncode}")
                if launch.healthy(port):
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("Server startup timed out")
            result["startup_seconds"] = round(time.monotonic() - started, 2)
            quick = post(port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Reply with one short word: ready"}],
                "max_tokens": 16, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            result["quick_inference"] = bool(quick.get("choices"))
            result["reasoning_control"] = test_reasoning_control(port)
            prompt, count = target_prompt(port, int(context * 0.75)) if args.context else ("Reply with one short word: ready", None)
            if count is None:
                count = token_count(port, prompt)
            result["measured_input_tokens"] = count
            result["window_fraction"] = round(count / context, 3)
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 16,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            inference_start = time.monotonic()
            response = post(port, "/v1/chat/completions", body)
            result["inference_seconds"] = round(time.monotonic() - inference_start, 2)
            result["usage"] = response.get("usage")
            result["finish_reason"] = response["choices"][0].get("finish_reason")
            control = result["reasoning_control"]
            result["passed"] = (result["quick_inference"] and bool(response["choices"]) and count < context
                                and control["id_received"] and control["control_success"] and control["final_content"])
        except (OSError, RuntimeError, KeyError, ValueError, KeyboardInterrupt) as error:
            result["passed"] = False
            result["error"] = str(error)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            signal.signal(signal.SIGTERM, old_sigterm)
    destination = logs / "validation.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Validation report: {destination}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
