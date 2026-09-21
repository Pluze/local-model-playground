"""Storage inventory, diagnostics, and conservative cleanup."""

from __future__ import annotations

import argparse
import errno
import os
import platform
import shutil
import sys
import time
from pathlib import Path

from playground.image import launcher as image_launcher
from playground.llm import assets as llm_assets
from playground.paths import (ASSETS_ROOT, HISTORY_ROOT, LOCAL_ROOT, MODELS_ROOT,
                              PROJECT_ROOT, RUNTIMES_ROOT, WORK_ROOT,
                              ensure_local_layout)


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                pass
    return total


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def storage_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Show local model-playground disk usage.")
    parser.parse_args(argv)
    ensure_local_layout()
    rows = []
    for name, directory in (("assets", ASSETS_ROOT), ("history", HISTORY_ROOT),
                            ("work", WORK_ROOT)):
        rows.append((name, tree_size(directory)))
    print(f"Local data: {LOCAL_ROOT}")
    for name, size in rows:
        print(f"  {name:12} {human_size(size):>10}")
    print(f"  {'total':12} {human_size(sum(size for _, size in rows)):>10}")
    usage = shutil.disk_usage(LOCAL_ROOT)
    print(f"Free space: {human_size(usage.free)}")
    return 0


def doctor_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the local toolchain and installed assets.")
    parser.add_argument("--full", action="store_true", help="hash every installed model file")
    args = parser.parse_args(argv)
    ensure_local_layout()
    okay = True

    checks = [
        ("Python >= 3.9", sys.version_info >= (3, 9), platform.python_version()),
        ("Local data writable", os.access(LOCAL_ROOT, os.W_OK), str(LOCAL_ROOT)),
    ]
    if platform.system() == "Darwin":
        checks.append(("Apple Silicon", platform.machine().lower() in ("arm64", "aarch64"), platform.machine()))
    for label, passed, detail in checks:
        print(f"{'OK' if passed else 'FAIL':4}  {label:22} {detail}")
        okay &= passed

    print("\nInstalled models:")
    for key in llm_assets.profile_names():
        _, spec = llm_assets.profile(key)
        entries = [spec["model"]] + ([spec["adapter"]] if spec.get("adapter") else [])
        valid = True
        for entry in entries:
            path = LOCAL_ROOT / entry["path"]
            valid &= path.is_file() and path.stat().st_size == entry["size"]
            if valid and args.full:
                valid &= llm_assets.verify_file(path, entry)
        print(f"  {'OK' if valid else '--':4} llm/{key:12} {len(entries)} component(s)")
    for key, model in image_launcher.MODELS.items():
        present = []
        valid = True
        for spec in model["assets"].values():
            path = LOCAL_ROOT / spec["path"]
            present.append(path.is_file())
            valid &= path.is_file() and path.stat().st_size == spec["size"]
            if valid and args.full:
                valid &= image_launcher.verified(path, spec, full=True)
        print(f"  {'OK' if valid else '--':4} image/{key:10} {sum(present)}/{len(present)} components")

    _, _, _, _, llm_server = llm_assets.paths("compact")
    runtimes = [
        ("llama.cpp", llm_server),
        ("ComfyUI", image_launcher.RUNTIME_PYTHON),
    ]
    print("\nRuntimes:")
    for name, binary in runtimes:
        present = binary.is_file() and os.access(binary, os.X_OK)
        print(f"  {'OK' if present else '--':4} {name:22} {binary}")
    print("\nDoctor result: toolchain ready." if okay else "\nDoctor result: required tools are missing.")
    return 0 if okay else 1


def _remove(path: Path) -> int:
    size = tree_size(path)
    if path.is_dir() and not path.is_symlink():
        for attempt in range(4):
            try:
                shutil.rmtree(path)
                break
            except OSError as error:
                if error.errno != errno.ENOTEMPTY or attempt == 3:
                    raise
                time.sleep(0.1 * (attempt + 1))
    else:
        path.unlink(missing_ok=True)
    return size


def clean_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Remove disposable local model-playground data.")
    parser.add_argument("--models", action="store_true", help="also remove downloaded model weights")
    parser.add_argument("--runtimes", action="store_true", help="also remove downloaded/built runtimes")
    parser.add_argument("--all", action="store_true", help="remove the complete .local directory")
    parser.add_argument("--yes", action="store_true", help="skip confirmation for model/runtime removal")
    args = parser.parse_args(argv)
    ensure_local_layout()
    targets = [HISTORY_ROOT, WORK_ROOT]
    for cache in PROJECT_ROOT.rglob("__pycache__"):
        if ".local" not in cache.parts:
            targets.append(cache)
    targets.extend(LOCAL_ROOT.rglob("*.part"))
    targets.extend(LOCAL_ROOT.rglob("*.invalid-*"))
    targets.extend(LOCAL_ROOT.rglob(".DS_Store"))
    if args.all:
        targets = [LOCAL_ROOT]
    else:
        if args.models:
            targets.append(MODELS_ROOT)
        if args.runtimes:
            targets.append(RUNTIMES_ROOT)
    destructive = args.all or args.models or args.runtimes
    if destructive and not args.yes:
        answer = input("This will remove downloaded assets. Continue? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cleanup cancelled.")
            return 1
    freed = 0
    seen = set()
    for target in targets:
        try:
            resolved = target.resolve()
        except OSError:
            continue
        if resolved in seen or not target.exists():
            continue
        seen.add(resolved)
        freed += _remove(target)
    print(f"Removed disposable data; reclaimed about {human_size(freed)}.")
    return 0
