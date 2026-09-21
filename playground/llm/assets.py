#!/usr/bin/env python3
"""Install and verify pinned text-model assets."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from playground.paths import DOWNLOADS_ROOT, LOCAL_ROOT, PROJECT_ROOT, RUNTIMES_ROOT

ROOT = PROJECT_ROOT
ASSET_ROOT = LOCAL_ROOT
LOCK = json.loads((Path(__file__).resolve().parent / "assets.lock.json").read_text(encoding="utf-8"))
HF_TOKEN_PATHS = (Path.home() / ".cache" / "huggingface" / "token",
                  Path.home() / ".huggingface" / "token")
SESSION_HF_TOKEN = None
KEYCHAIN_SERVICE = "local-model-playground.huggingface"
KEYCHAIN_ACCOUNT = "huggingface"


def platform_key():
    systems = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}
    machines = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "amd64": "x64"}
    key = f"{systems.get(platform.system(), platform.system().lower())}-{machines.get(platform.machine().lower(), platform.machine().lower())}"
    return key


def profile_names():
    return tuple(LOCK["profiles"])


def default_profile():
    return os.environ.get("LOCAL_LLM_PROFILE", LOCK["default_profile"])


def profile(profile=None):
    name = profile or default_profile()
    try:
        return name, LOCK["profiles"][name]
    except KeyError:
        choices = ", ".join(profile_names())
        raise RuntimeError(f"Unknown model profile '{name}'. Choose one of: {choices}.") from None


def profile_choices():
    return tuple((name, f"{spec['label']} - {spec['description']}")
                 for name, spec in LOCK["profiles"].items())


def paths(profile_name=None):
    key = platform_key()
    if key not in LOCK["runtime"] and not os.environ.get("LOCAL_LLM_RUNTIME"):
        raise RuntimeError(f"No pinned PrismML runtime for {key}. Set LOCAL_LLM_RUNTIME to a compatible directory.")
    _, selected = profile(profile_name)
    custom_model = os.environ.get("LOCAL_LLM_MODEL")
    model = Path(custom_model or str(ASSET_ROOT / selected["model"]["path"])).expanduser()
    if "LOCAL_LLM_ADAPTER" in os.environ:
        adapter_value = os.environ["LOCAL_LLM_ADAPTER"].strip()
        adapter = Path(adapter_value).expanduser() if adapter_value else None
    elif custom_model:
        adapter = None
    elif selected.get("adapter"):
        adapter = ASSET_ROOT / selected["adapter"]["path"]
    else:
        adapter = None
    runtime = Path(os.environ.get(
        "LOCAL_LLM_RUNTIME", str(RUNTIMES_ROOT / "llama" / key)
    )).expanduser()
    server = runtime / ("llama-server.exe" if os.name == "nt" else "llama-server")
    return key, model, adapter, runtime, server


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, spec):
    return (path.is_file() and ("size" not in spec or path.stat().st_size == spec["size"])
            and sha256(path) == spec["sha256"])


def check(profile_name=None):
    profile_name, selected = profile(profile_name)
    key, model, adapter, runtime, server = paths(profile_name)
    issues = []
    if not os.environ.get("LOCAL_LLM_MODEL") and not verify_file(model, selected["model"]):
        issues.append(f"model: missing or checksum mismatch: {model}")
    elif os.environ.get("LOCAL_LLM_MODEL") and not model.is_file():
        issues.append(f"model: missing: {model}")
    if adapter is not None:
        if "LOCAL_LLM_ADAPTER" in os.environ:
            if not adapter.is_file():
                issues.append(f"adapter: missing: {adapter}")
        elif not verify_file(adapter, selected["adapter"]):
            issues.append(f"adapter: missing or checksum mismatch: {adapter}")
    marker = runtime / ".runtime-sha256"
    if not server.is_file() or (not os.environ.get("LOCAL_LLM_RUNTIME") and
            (not marker.is_file() or marker.read_text().strip() != LOCK["runtime"][key]["sha256"])):
        issues.append(f"runtime: missing or version mismatch: {runtime}")
    print(f"Platform: {key}")
    print(f"Profile: {profile_name} ({selected['label']})")
    print(f"Model: {model}")
    print(f"Adapter: {adapter or 'none'}")
    print(f"Runtime: {runtime}")
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return False
    print("All pinned assets verified.")
    return True


def huggingface_token():
    token = SESSION_HF_TOKEN or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT,
                 "-s", KEYCHAIN_SERVICE, "-w"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    for path in HF_TOKEN_PATHS:
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            return token
    return None


def set_session_huggingface_token(token):
    global SESSION_HF_TOKEN
    SESSION_HF_TOKEN = token.strip() or None


def download(url, target, spec):
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if verify_file(target, spec):
        print(f"Verified: {target}")
        return
    if target.exists():
        target.unlink()
    token = huggingface_token()
    if spec.get("gated") and not token:
        raise RuntimeError(
            f"Access to this model requires accepting its Hugging Face terms at {spec['source']} "
            "and setting HF_TOKEN for the first download."
        )
    for attempt in range(2):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Local-Uncensored-LLM-All-in-One/1"}
        if token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {token}"
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with part.open(mode) as output:
                    total = offset if append else 0
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        total += len(block)
                        print(f"\rDownloading {target.name}: {total / 1e9:.2f} GB", end="", flush=True)
        except urllib.error.HTTPError as error:
            if spec.get("gated") and error.code in (401, 403):
                raise RuntimeError(
                    f"Hugging Face denied access to this model. Accept its terms at {spec['source']} "
                    "and provide an authorized HF_TOKEN."
                ) from error
            raise
        print()
        if verify_file(part, spec):
            os.replace(part, target)
            print(f"Verified: {target}")
            return
        part.unlink(missing_ok=True)
        if attempt == 0:
            print("Checksum mismatch; retrying from the beginning.", file=sys.stderr)
    raise RuntimeError(f"Checksum mismatch after retry: {target}")


def safe_member(name, link=None):
    parts = Path(name).parts
    if Path(name).is_absolute() or ".." in parts:
        raise RuntimeError(f"Unsafe archive path: {name}")
    if link is not None and (Path(link).is_absolute() or ".." in Path(link).parts):
        raise RuntimeError(f"Unsafe archive link: {name} -> {link}")


def extract_runtime(archive, destination):
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    safe_member(member.filename)
                    if stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK:
                        raise RuntimeError("Archive contains unsupported ZIP links")
                package.extractall(stage)
        else:
            with tarfile.open(archive, "r:gz") as package:
                for member in package.getmembers():
                    safe_member(member.name, member.linkname if member.issym() or member.islnk() else None)
                    if member.isdev() or member.isfifo():
                        raise RuntimeError("Archive contains a special device")
                package.extractall(stage)
        candidates = list(stage.rglob("llama-server.exe" if os.name == "nt" else "llama-server"))
        if len(candidates) != 1:
            raise RuntimeError("Runtime archive does not contain one llama-server executable")
        package_dir = candidates[0].parent
        shutil.move(str(package_dir), str(destination))
    if os.name != "nt":
        server = destination / "llama-server"
        server.chmod(server.stat().st_mode | stat.S_IXUSR)


def install(profile_name=None):
    profile_name, selected = profile(profile_name)
    key, model, adapter, runtime, server = paths(profile_name)
    if os.environ.get("LOCAL_LLM_MODEL"):
        if not model.is_file():
            raise RuntimeError(f"Custom model is missing: {model}")
        print(f"Using custom model: {model}")
    else:
        download(selected["model"]["url"], model, selected["model"])
    if adapter is not None:
        if "LOCAL_LLM_ADAPTER" in os.environ:
            if not adapter.is_file():
                raise RuntimeError(f"Custom adapter is missing: {adapter}")
            print(f"Using custom adapter: {adapter}")
        else:
            download(selected["adapter"]["url"], adapter, selected["adapter"])
    if os.environ.get("LOCAL_LLM_RUNTIME"):
        if not server.is_file():
            raise RuntimeError(f"Custom runtime is missing {server}")
        print(f"Using custom runtime: {runtime}")
    else:
        spec = LOCK["runtime"][key]
        marker = runtime / ".runtime-sha256"
        if not server.is_file() or not marker.is_file() or marker.read_text().strip() != spec["sha256"]:
            runtime.parent.mkdir(parents=True, exist_ok=True)
            release = LOCK["runtime_release"]
            extension = "zip" if key.startswith("windows") else "tar.gz"
            filename = f"llama-{release}-bin-{spec['asset']}.{extension}"
            url = f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{release}/{filename}"
            archive = DOWNLOADS_ROOT / filename
            download(url, archive, {"sha256": spec["sha256"]})
            if runtime.exists():
                shutil.rmtree(runtime)
            extract_runtime(archive, runtime)
            marker.write_text(spec["sha256"] + "\n", encoding="utf-8")
            archive.unlink()
    if not check(profile_name):
        raise RuntimeError("Installation verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing files without downloading")
    parser.add_argument("--model", "--profile", choices=profile_names(), default=default_profile(),
                        help="model profile to install or verify")
    args = parser.parse_args()
    try:
        if args.check:
            return 0 if check(args.model) else 1
        install(args.model)
        return 0
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
