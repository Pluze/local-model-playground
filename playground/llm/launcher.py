#!/usr/bin/env python3
"""Run one text model for the lifetime of this CLI."""
import argparse
import getpass
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from http.server import ThreadingHTTPServer

from playground.llm.proxy import RollingMemory, make_handler, output_reserve
from playground.llm import assets as install
from playground.paths import HISTORY_ROOT, LOGS_ROOT, PROJECT_ROOT, STATE_ROOT

ROOT = PROJECT_ROOT
LOG_LOCK = threading.Lock()
CONTEXT_PRESETS = ((4096, "Minimal (4K)"), (8192, "Compact (8K)"),
                   (16384, "Balanced (16K)"), (32768, "Extended (32K)"),
                   (65536, "Maximum (64K)"))
ACTIVE_PORT = STATE_ROOT / "llm-active-port"
HF_TOKEN_PAGE = "https://huggingface.co/settings/tokens"


def prompt_choice(title, choices, default):
    print(f"\n{title}:")
    for index, (value, label) in enumerate(choices, 1):
        marker = " (default)" if value == default else ""
        print(f"  {index}. {label}{marker}")
    while True:
        answer = input("Select a number or press Enter for the default: ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1][0]
        print(f"Choose a number from 1 to {len(choices)}.")


def prompt_yes_no(question):
    while True:
        answer = input(f"{question} [y/N]: ").strip().lower()
        if not answer or answer in ("n", "no"):
            return False
        if answer in ("y", "yes"):
            return True
        print("Enter y or n.")


def prepare_model_access(profile_name):
    _, selected = install.profile(profile_name)
    spec = selected["model"]
    if not spec.get("gated"):
        return True
    _, model, _, _, _ = install.paths(profile_name)
    if model.is_file() or install.huggingface_token():
        return True
    source = spec["source"]
    print("\nThis model requires one-time Hugging Face account approval before it can be downloaded.")
    print(f"  1. Sign in and accept the model conditions: {source}")
    print(f"  2. Create or copy a read token: {HF_TOKEN_PAGE}")
    print("Opening both pages in your browser. Return here after completing those steps.")
    webbrowser.open(source, new=2)
    webbrowser.open(HF_TOKEN_PAGE, new=2)
    token = getpass.getpass("Paste the Hugging Face read token (hidden), or press Enter to choose another model: ").strip()
    if not token:
        print("No token entered. Returning to model selection.")
        return False
    install.set_session_huggingface_token(token)
    print("Token accepted for this launch. It is not saved by the launcher.")
    return True


def port_busy(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def existing_proxy(port=None):
    if port is None:
        try:
            recorded = int(ACTIVE_PORT.read_text())
            ports = (recorded, 8080) if 0 < recorded < 65536 else (8080,)
        except (OSError, ValueError):
            ports = (8080,)
    else:
        ports = (port,)
    for candidate in ports:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{candidate}/orca/status", timeout=0.5) as response:
                status = json.load(response)
            if isinstance(status, dict) and isinstance(status.get("active"), bool) and "started" in status:
                status.update({"url": f"http://127.0.0.1:{candidate}", "port": candidate})
                return status
        except (OSError, ValueError, urllib.error.URLError):
            pass
    return None


def listener_pid(port):
    """Find the proxy owner for services started before status exposed its PID."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid > 1:
            return pid
    return None


def stop_existing_instance(instance, timeout=20):
    pid = instance.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        pid = listener_pid(instance["port"])
    if pid is None:
        raise RuntimeError("Could not identify the existing launcher's process. Close its terminal and try again.")
    if pid == os.getpid():
        raise RuntimeError("Refusing to stop the current launcher process.")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise RuntimeError(f"Permission denied while stopping launcher process {pid}.") from error
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if existing_proxy(instance["port"]) is None:
            print(f"Stopped launcher process {pid}.", flush=True)
            return
        time.sleep(0.2)
    raise RuntimeError(f"Launcher process {pid} did not stop within {timeout} seconds.")


def interrupt(signum, frame):
    raise KeyboardInterrupt


def start_backend(command, cwd, log):
    backend_environment = os.environ.copy()
    backend_environment.pop("HF_TOKEN", None)
    backend_environment.pop("HUGGING_FACE_HUB_TOKEN", None)
    isolation = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                 if os.name == "nt" else {"start_new_session": True})
    process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace", bufsize=1,
                               env=backend_environment, **isolation)

    def relay():
        with process.stdout:
            for line in process.stdout:
                with LOG_LOCK:
                    log.write(line)
                    log.flush()
                    print(f"[model] {line}", end="", flush=True)

    process.log_thread = threading.Thread(target=relay, daemon=True)
    process.log_thread.start()
    return process


def profile_defaults(profile_name=None):
    """Return one model's launch defaults from the pinned manifest."""
    _, selected = install.profile(profile_name)
    return selected["defaults"]


def build_command(server, model, adapter, backend_port, context, profile_name=None):
    defaults = profile_defaults(profile_name)
    gpu_default = "99" if platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64") else "0"
    parallel = os.environ.get("LOCAL_LLM_PARALLEL", str(defaults["parallel"]))
    kv_cache = os.environ.get("LOCAL_LLM_KV_CACHE", defaults["kv_cache"])
    command = [
        str(server), "-m", str(model),
        "--host", "127.0.0.1", "--port", str(backend_port),
        "-c", str(context),
        "-np", parallel,
        "-ngl", os.environ.get("LOCAL_LLM_GPU_LAYERS", gpu_default),
        "-fa", "on", "-ctk", kv_cache, "-ctv", kv_cache,
        "--cache-ram", "512", "--jinja",
        "--temp", str(defaults["temperature"]), "--top-p", str(defaults["top_p"]),
        "--top-k", str(defaults["top_k"]),
    ]
    if adapter is not None:
        command[3:3] = ["--lora", str(adapter)]
    return command


def window_settings(context, parallel, compact_at=None, keep_tokens=None):
    if context < 4096 or parallel < 1 or context // parallel < 4096:
        raise ValueError("Context must provide at least 4096 tokens per parallel slot.")
    slot_context = context // parallel
    reserve = output_reserve(slot_context)
    if compact_at is None:
        compact_at = min(int(slot_context * 0.85), slot_context - reserve)
    if keep_tokens is None:
        keep_tokens = min(int(slot_context * 0.37), compact_at // 2)
    if not (0 < keep_tokens < compact_at <= slot_context - reserve):
        raise ValueError(f"Require 0 < keep tokens < compact at <= per-slot context - {reserve}.")
    return slot_context, compact_at, keep_tokens


def stop_backend(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if process is not None and hasattr(process, "log_thread"):
        process.log_thread.join(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-prompt", action="store_true", help="use defaults without interactive questions")
    parser.add_argument("--check", action="store_true", help="verify assets and exit without starting services")
    parser.add_argument("--no-browser", action="store_true", help="print the UI URL without opening a browser")
    parser.add_argument("--model", "--profile", choices=install.profile_names(),
                        help="model profile (default: interactive selection or compact)")
    parser.add_argument("--context", "--ctx", type=int,
                        help="context window in tokens (default: selected model's balanced preset)")
    parser.add_argument("--compact-at", type=int,
                        help="start summarizing at this many input tokens (default: about 85%% of context)")
    parser.add_argument("--keep-tokens", type=int,
                        help="retain this many recent tokens after summarizing (default: about 37%% of context)")
    args = parser.parse_args(argv)
    if not args.check:
        running = existing_proxy()
        if running:
            pid_text = f" (process {running['pid']})" if isinstance(running.get("pid"), int) else ""
            print(f"A playground text model is already running at {running['url']}{pid_text}.")
            if args.no_prompt or not sys.stdin.isatty():
                print("Run interactively to be asked whether to stop it.")
                return 0
            try:
                if not prompt_yes_no("Stop the existing service and continue"):
                    return 0
                stop_existing_instance(running)
            except (EOFError, KeyboardInterrupt):
                print("\nStartup cancelled.", file=sys.stderr)
                return 1
            except RuntimeError as error:
                print(error, file=sys.stderr)
                return 1
    try:
        profile_name = args.model or install.default_profile()
        defaults = profile_defaults(profile_name)
        context = args.context if args.context is not None else int(os.environ.get("LOCAL_LLM_CTX", defaults["context"]))
        args.compact_at = args.compact_at if args.compact_at is not None else (
            int(os.environ["LOCAL_LLM_COMPACT_AT"]) if "LOCAL_LLM_COMPACT_AT" in os.environ else None)
        args.keep_tokens = args.keep_tokens if args.keep_tokens is not None else (
            int(os.environ["LOCAL_LLM_KEEP_TOKENS"]) if "LOCAL_LLM_KEEP_TOKENS" in os.environ else None)
        interactive = sys.stdin.isatty() and not args.check and not args.no_prompt
        if interactive:
            if args.model is None and "LOCAL_LLM_PROFILE" not in os.environ:
                default_profile_name = profile_name
                while True:
                    profile_name = prompt_choice("Model", install.profile_choices(), profile_name)
                    if prepare_model_access(profile_name):
                        break
                    profile_name = default_profile_name
                defaults = profile_defaults(profile_name)
                if args.context is None and "LOCAL_LLM_CTX" not in os.environ:
                    context = defaults["context"]
            elif not prepare_model_access(profile_name):
                print("Startup cancelled.", file=sys.stderr)
                return 1
            print("Choose the context window for this launch (press Enter for the default).")
            if args.context is None and "LOCAL_LLM_CTX" not in os.environ:
                context = prompt_choice("Context window", CONTEXT_PRESETS, context)
    except (EOFError, KeyboardInterrupt):
        print("\nStartup cancelled.", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    try:
        parallel = int(os.environ.get("LOCAL_LLM_PARALLEL", str(defaults["parallel"])))
    except ValueError:
        parser.error("LOCAL_LLM_PARALLEL must be a positive integer")
    if parallel < 1:
        parser.error("LOCAL_LLM_PARALLEL must be a positive integer")
    try:
        slot_context, compact_at, keep_tokens = window_settings(context, parallel, args.compact_at, args.keep_tokens)
    except ValueError as error:
        parser.error(str(error))
    if args.check:
        return 0 if install.check(profile_name) else 1
    try:
        install.install(profile_name)
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    _, model, adapter, _, server = install.paths(profile_name)
    port = int(os.environ.get("LOCAL_LLM_PORT", "0")) or (8080 if not port_busy(8080) else free_port())
    backend_port = int(os.environ.get("LOCAL_LLM_BACKEND_PORT", "0")) or (8081 if not port_busy(8081) else free_port())
    while backend_port == port:
        backend_port = free_port()
    if port == backend_port or any(port_busy(candidate) for candidate in (port, backend_port)):
        print("The UI and backend need separate, unused ports. Set LOCAL_LLM_PORT or LOCAL_LLM_BACKEND_PORT.", file=sys.stderr)
        return 1
    command = build_command(server, model, adapter, backend_port, context, profile_name)
    _, selected_profile = install.profile(profile_name)
    print(f"Model: {selected_profile['label']} ({profile_name}); adapter: {adapter or 'none'}", flush=True)
    print(f"Context: {context} tokens ({slot_context} per slot, {parallel} slot(s)); "
          f"compact at: {compact_at}; keep recent: {keep_tokens}", flush=True)
    for name in ("SIGHUP", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, interrupt)

    logs = LOGS_ROOT / "llm"
    logs.mkdir(exist_ok=True)
    process = None
    httpd = None
    url = f"http://127.0.0.1:{port}"
    with (logs / "server.log").open("w") as log:
        try:
            process = start_backend(command, server.parent, log)
            print("Starting Local Model Playground text backend...", flush=True)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {process.returncode}. See {logs / 'server.log'}")
                if healthy(backend_port):
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError(f"Server startup timed out. See {logs / 'server.log'}")
            memory = RollingMemory(HISTORY_ROOT / "llm", backend_port, slot_context, compact_at, keep_tokens)
            httpd = ThreadingHTTPServer(("127.0.0.1", port),
                                        make_handler(memory))
            httpd.daemon_threads = True
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            ACTIVE_PORT.parent.mkdir(exist_ok=True)
            ACTIVE_PORT.write_text(str(port))
            if not args.no_browser and webbrowser.open(url, new=2):
                print(f"Browser opened: {url}", flush=True)
            else:
                print(f"Open this URL in your browser: {url}", flush=True)
            print("Keep this CLI open while chatting. Press Ctrl+C or close it to stop the server.", flush=True)
            while True:
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(f"Server exited unexpectedly with code {returncode}. See {logs / 'server.log'}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping the server...", flush=True)
        except (RuntimeError, OSError, ValueError) as exc:
            print(exc, file=sys.stderr, flush=True)
            return 1
        finally:
            if httpd is not None:
                httpd.shutdown()
                httpd.server_close()
            try:
                if ACTIVE_PORT.read_text() == str(port):
                    ACTIVE_PORT.unlink()
            except OSError:
                pass
            stop_backend(process)
    return 0


if __name__ == "__main__":
    sys.exit(main())
