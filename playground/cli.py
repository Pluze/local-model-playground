"""Unified command line for running exactly one local model at a time."""

import json
import os
import sys
import time

from playground import __version__
from playground import benchmark
from playground.image import launcher as image_launcher
from playground.llm import launcher as llm_launcher
from playground.llm import verifier as llm_verifier
from playground.paths import STATE_ROOT, ensure_local_layout
from playground.storage import clean_main, doctor_main, storage_main


SESSION = STATE_ROOT / "session.json"
MODELS = (
    ("llm", "compact", "Compact LLM — about 6 GB, lowest memory"),
    ("llm", "uncensored", "Uncensored LLM — about 13.5 GB, HF approval required"),
    ("llm", "direct", "Direct LLM — about 17 GB, highest quality/memory"),
    ("image", "flux2", "FLUX.2 image — about 5.3 GB, recommended"),
    ("image", "qwen21", "Qwen-Image-2.1 — about 14.2 GB, quality experiments"),
)


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return int(pid) > 1
    except (OSError, TypeError, ValueError):
        return False


def _claim_session(kind: str, model: str) -> bool:
    ensure_local_layout()
    try:
        current = json.loads(SESSION.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        current = None
    if current and _pid_alive(current.get("pid")):
        print(
            f"Another playground model is active: {current.get('kind')}/{current.get('model')} "
            f"(process {current.get('pid')}). Stop it before launching another model.",
            file=sys.stderr,
        )
        return False
    SESSION.write_text(
        json.dumps({"pid": os.getpid(), "kind": kind, "model": model, "started": time.time()}),
        encoding="utf-8",
    )
    return True


def _release_session() -> None:
    try:
        state = json.loads(SESSION.read_text(encoding="utf-8"))
        if state.get("pid") == os.getpid():
            SESSION.unlink()
    except (OSError, ValueError, json.JSONDecodeError):
        pass


def _run(kind: str, model: str, args: list[str]) -> int:
    if not _claim_session(kind, model):
        return 1
    try:
        forwarded = ["--model", model, *args]
        if kind == "llm":
            return llm_launcher.main(forwarded)
        return image_launcher.main(forwarded)
    finally:
        _release_session()


def _interactive() -> int:
    if not sys.stdin.isatty():
        print_help()
        return 2
    print("\nLocal Model Playground\nRun one model at a time; Ctrl+C returns all memory to the system.\n")
    for index, (_, _, label) in enumerate(MODELS, 1):
        print(f"  {index}. {label}")
    print("  0. Exit")
    while True:
        answer = input("Select a model [1]: ").strip() or "1"
        if answer == "0":
            return 0
        if answer.isdigit() and 1 <= int(answer) <= len(MODELS):
            kind, model, _ = MODELS[int(answer) - 1]
            return _run(kind, model, [])
        print(f"Choose a number from 0 to {len(MODELS)}.")


def print_help() -> None:
    print(f"""Local Model Playground {__version__}

Run one model at a time from a shared local cache.

Usage:
  python3 launch.py                         Interactive model picker
  python3 launch.py llm [LLM options]       Launch a text model
  python3 launch.py image [image options]   Launch an image model
  python3 launch.py verify [options]        Run isolated LLM inference checks
  python3 launch.py benchmark [options]     Benchmark one model with recommended defaults
  python3 launch.py doctor [--full]         Check tools and local assets
  python3 launch.py storage                 Show local disk usage
  python3 launch.py clean [options]         Remove disposable local data

Examples:
  python3 launch.py llm --model compact --context 8192
  python3 launch.py image --model flux2 --no-browser
  python3 launch.py clean

All large/generated files live under .local/ by default. Set
MODEL_PLAYGROUND_HOME to place that cache on another disk.
""")


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return _interactive()
    command, tail = args[0], args[1:]
    if command in ("-h", "--help", "help"):
        print_help()
        return 0
    if command in ("-V", "--version"):
        print(__version__)
        return 0
    if command == "llm":
        model = "compact"
        if "--model" in tail:
            try:
                model = tail[tail.index("--model") + 1]
            except IndexError:
                pass
        return _run("llm", model, tail)
    if command == "image":
        model = "flux2"
        if "--model" in tail:
            try:
                model = tail[tail.index("--model") + 1]
            except IndexError:
                pass
        return _run("image", model, tail)
    if command == "verify":
        return llm_verifier.main(tail)
    if command == "benchmark":
        return benchmark.main(tail)
    if command == "doctor":
        return doctor_main(tail)
    if command == "storage":
        return storage_main(tail)
    if command == "clean":
        return clean_main(tail)
    for kind, model, _ in MODELS:
        if command == model:
            return _run(kind, model, tail)
    print(f"Unknown command or model: {command}\n", file=sys.stderr)
    print_help()
    return 2
