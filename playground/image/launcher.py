#!/usr/bin/env python3
"""Automatically install and run one local image model in ComfyUI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from playground.paths import (CACHE_ROOT, HISTORY_ROOT, LOCAL_ROOT, LOGS_ROOT,
                              RUNTIMES_ROOT, STATE_ROOT, WORK_ROOT)


VERSION = "0.2.0"
RUNTIME_SOURCE = RUNTIMES_ROOT / "comfyui"
RUNTIME_PYTHON = RUNTIMES_ROOT / "comfyui-venv" / "bin" / "python"
UV_BINARY = RUNTIMES_ROOT / "uv" / "uv"
RUNTIME_MARKER = RUNTIME_SOURCE / ".playground-runtime.json"
MODELS_DIRECTORY = WORK_ROOT / "comfyui-models"
USER_DIRECTORY = HISTORY_ROOT / "comfyui-user"
OUTPUT_DIRECTORY = HISTORY_ROOT / "images"
INPUT_DIRECTORY = HISTORY_ROOT / "input"
TEMP_DIRECTORY = CACHE_ROOT / "comfyui"
ACTIVE_STATE = STATE_ROOT / "image-active.json"
LOG_LOCK = threading.Lock()

COMFY_COMMIT = "b0f4b7b294ce482a2e071d9d762c133d38c7aa07"
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
RUNTIME_FILES = {
    "uv": {
        "path": "assets/downloads/uv-0.12.17-aarch64-apple-darwin.tar.gz",
        "url": "https://github.com/astral-sh/uv/releases/download/0.12.17/uv-aarch64-apple-darwin.tar.gz",
        "size": 16929004,
        "sha256": "85f00cbdc6dd3e97eba4c31b4d014375a9fdfe8f570023b84e5102fc3456896b",
    },
    "comfyui": {
        "path": f"assets/downloads/comfyui-{COMFY_COMMIT}.tar.gz",
        "url": f"https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/{COMFY_COMMIT}",
        "size": 12510322,
        "sha256": "1a48f6a6111d2992f75e7a379f2eff178fd55a347ce77a25597ecac6a125b859",
    },
    "gguf": {
        "path": f"assets/downloads/comfyui-gguf-{GGUF_COMMIT}.tar.gz",
        "url": f"https://codeload.github.com/city96/ComfyUI-GGUF/tar.gz/{GGUF_COMMIT}",
        "size": 31636,
        "sha256": "688126d5c2b8c4ff061f56b23c3c6791ac01dc366e98d41ef1aaef374c36f283",
    },
}


def hf_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true"


MODELS = {
    "flux2": {
        "label": "FLUX.2 Klein Base 4B (recommended)",
        "short_label": "FLUX.2 Klein Base 4B",
        "steps": 12,
        "cfg": 4.0,
        "presets": {
            "fast": {"short": "Fast", "label": "Fast preview", "steps": 8, "cfg": 4.0, "sampler": "euler", "scheduler": "flux2", "advice": "Use while iterating on composition; move to Balanced when the prompt is settled."},
            "balanced": {"short": "Balanced", "label": "Recommended balance", "steps": 12, "cfg": 4.0, "sampler": "euler", "scheduler": "flux2", "advice": "The default for this quantized model: good detail without paying for many low-value steps."},
            "detail": {"short": "Detail", "label": "More detail", "steps": 20, "cfg": 4.0, "sampler": "euler", "scheduler": "flux2", "advice": "Try for a final image. Above 20 steps usually costs more time than it adds quality."},
        },
        "sizes": [
            {"label": "1:1", "width": 512, "height": 512},
            {"label": "4:3", "width": 768, "height": 576},
            {"label": "3:4", "width": 576, "height": 768},
        ],
        "samplers": ["euler", "euler_ancestral", "dpmpp_2m"],
        "schedulers": ["flux2"],
        "parameter_advice": "Start near 8–20 steps and CFG 3–5. More pixels and steps increase latency and memory; change one control at a time.",
        "parameter_guides": {
            "steps": [
                {"max": 8, "label": "Fast preview", "text": "Faster iteration; composition is useful, but fine detail may be unfinished."},
                {"max": 20, "label": "Recommended: 8–20", "text": "Best speed/quality range; 12 is the balanced default."},
                {"max": 60, "label": "Diminishing returns", "text": "More time and heat, usually with only subtle detail gains."},
            ],
            "cfg": [
                {"max": 2.9, "label": "Loose guidance", "text": "More freedom, but the image may follow the prompt less precisely."},
                {"max": 5, "label": "Recommended: 3–5", "text": "Good prompt adherence without forcing the composition too hard."},
                {"max": 20, "label": "Strong guidance", "text": "May look harsh, oversaturated, or less natural."},
            ],
        },
        "option_guides": {
            "sampler": {
                "euler": {"label": "Recommended", "text": "Predictable and fast; the safest baseline for FLUX.2."},
                "euler_ancestral": {"label": "More variation", "text": "Adds stochastic texture and variety; results can feel livelier but less controlled."},
                "dpmpp_2m": {"label": "Smoother", "text": "Often favors smooth detail and refinement; compare at the same seed before adopting it."},
            },
            "scheduler": {
                "flux2": {"label": "Required / recommended", "text": "FLUX.2-native noise schedule adjusted for the selected size and step count."},
            },
        },
        "template": "image_flux2_klein_text_to_image.json",
        "assets": {
            "diffusion": {
                "path": "assets/models/image/flux2/flux-2-klein-base-4b-Q4_0.gguf",
                "url": hf_url("leejet/FLUX.2-klein-base-4B-GGUF", "d12671125306ca6b5f6db1b33ed4c80c8511a53f", "flux-2-klein-base-4b-Q4_0.gguf"),
                "size": 2460378560,
                "sha256": "c3a2854510677b7aa37dd7547d908c54889a76c6d6aa3ffe902fcaa092d1328b",
            },
            "text_encoder": {
                "path": "assets/models/image/flux2/Qwen3-4B-Q4_K_M.gguf",
                "url": hf_url("unsloth/Qwen3-4B-GGUF", "22c9fc8a8c7700b76a1789366280a6a5a1ad1120", "Qwen3-4B-Q4_K_M.gguf"),
                "size": 2497281312,
                "sha256": "f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a",
            },
            "vae": {
                "path": "assets/models/image/flux2/flux2-vae.safetensors",
                "url": hf_url("Comfy-Org/flux2-klein-4B", "5f526678002e43af5551dadb73ce2e8c91b43afe", "split_files/vae/flux2-vae.safetensors"),
                "size": 336211292,
                "sha256": "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
            },
        },
    },
    "qwen21": {
        "label": "Qwen-Image-2.1 (quality)",
        "short_label": "Qwen-Image-2.1",
        "steps": 25,
        "cfg": 1.0,
        "presets": {
            "fast": {"short": "Fast", "label": "Fast preview", "steps": 12, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "advice": "A quick composition check before committing to the quality preset."},
            "balanced": {"short": "Balanced", "label": "Recommended quality", "steps": 25, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "advice": "The upstream-style baseline for reliable quality. CFG 1 intentionally leaves negative prompting inactive."},
            "detail": {"short": "Detail", "label": "More refinement", "steps": 40, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "advice": "Use selectively for final output; gains after 25 steps can be subtle."},
        },
        "sizes": [
            {"label": "512", "width": 512, "height": 512},
            {"label": "768", "width": 768, "height": 768},
            {"label": "1024", "width": 1024, "height": 1024},
        ],
        "samplers": ["euler", "euler_ancestral", "dpmpp_2m"],
        "schedulers": ["simple", "normal", "karras"],
        "parameter_advice": "Keep CFG at 1 for the recommended flow. Raise it only when deliberately testing guidance; square sizes are required by this workflow.",
        "parameter_guides": {
            "steps": [
                {"max": 12, "label": "Fast preview", "text": "Useful for composition checks; texture and small details may still change."},
                {"max": 25, "label": "Recommended: 12–25", "text": "25 is the quality baseline and the best default for normal tests."},
                {"max": 40, "label": "High detail", "text": "Slower refinement for selected final images."},
                {"max": 60, "label": "Diminishing returns", "text": "Large time cost with uncertain visible improvement."},
            ],
            "cfg": [
                {"max": 0.9, "label": "Very loose guidance", "text": "Prompt adherence can weaken."},
                {"max": 1, "label": "Recommended: 1", "text": "Native baseline; negative prompting is intentionally inactive."},
                {"max": 2, "label": "Experimental guidance", "text": "Stronger prompt pressure; negative prompting becomes relevant."},
                {"max": 20, "label": "Likely over-guided", "text": "Can reduce natural detail or introduce artifacts."},
            ],
        },
        "option_guides": {
            "sampler": {
                "euler": {"label": "Recommended", "text": "Direct, predictable baseline used by the official workflow."},
                "euler_ancestral": {"label": "More variation", "text": "Introduces fresh noise during sampling for more texture and less deterministic-looking detail."},
                "dpmpp_2m": {"label": "Smoother", "text": "A multistep solver that can improve smooth detail, especially with Normal or Karras scheduling."},
            },
            "scheduler": {
                "simple": {"label": "Recommended", "text": "Uses the model's native schedule directly; best baseline for Qwen-Image-2.1."},
                "normal": {"label": "Standard spacing", "text": "Even conventional sigma progression; useful when comparing alternate samplers."},
                "karras": {"label": "More finishing emphasis", "text": "Allocates more work to low-noise refinement; can improve fine detail but changes the look."},
            },
        },
        "template": "image_qwen_image_2_1_t2i.json",
        "native_diffusion": True,
        "native_text_encoder": True,
        "assets": {
            "diffusion": {
                "path": "assets/models/image/qwen21/qwen_image_2.1_int8_convrot.safetensors",
                "url": hf_url("Comfy-Org/Qwen-Image-2.1", "ace0edeb3791a594ddfa36ed5f41a178a394e921", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
                "size": 7256783064,
                "sha256": "cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d",
            },
            "text_encoder": {
                "path": "assets/models/image/qwen21/qwen3vl_8b_w4a8.safetensors",
                "url": hf_url("Comfy-Org/Qwen-Image-2.1", "ace0edeb3791a594ddfa36ed5f41a178a394e921", "text_encoders/qwen3vl_8b_w4a8.safetensors"),
                "size": 6312105364,
                "sha256": "7754425e55e7bea2bfde4dde59a4cc236cb44e5ee9c215ea66ef8d47012824eb",
            },
            "vae": {
                "path": "assets/models/image/qwen21/qwen_image_2.1_vae_bf16.safetensors",
                "url": hf_url("Comfy-Org/Qwen-Image-2.1", "ace0edeb3791a594ddfa36ed5f41a178a394e921", "vae/qwen_image_2.1_vae_bf16.safetensors"),
                "size": 675509688,
                "sha256": "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9",
            },
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, spec: dict, *, full: bool = False) -> bool:
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    marker = path.with_name(path.name + ".sha256")
    if not full and marker.is_file() and marker.read_text(encoding="utf-8").strip() == spec["sha256"]:
        return True
    actual = sha256(path)
    if actual != spec["sha256"]:
        return False
    marker.write_text(actual + "\n", encoding="utf-8")
    return True


def download(name: str, spec: dict) -> Path:
    target = LOCAL_ROOT / spec["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if verified(target, spec):
        print(f"Verified: {target}")
        return target
    target.unlink(missing_ok=True)
    part = target.with_name(target.name + ".part")
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Local-Model-Playground/0.2"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(f"Downloading {name}: {target.name}")
        request = urllib.request.Request(spec["url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                total = offset if append else 0
                report_at = total + 256 * 1024 * 1024
                with part.open("ab" if append else "wb") as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        total += len(block)
                        if total >= report_at or total >= spec["size"]:
                            print(f"\r  {total / 1e9:.2f}/{spec['size'] / 1e9:.2f} GB ({min(100, total * 100 / spec['size']):.1f}%)", end="", flush=True)
                            report_at = total + 256 * 1024 * 1024
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            if attempt == 3:
                raise
            print(f"\nDownload interrupted ({error}); resuming ({attempt + 2}/4)...", file=sys.stderr)
            continue
        print()
        if verified(part, spec, full=True):
            part.replace(target)
            part.with_name(part.name + ".sha256").replace(target.with_name(target.name + ".sha256"))
            return target
        part.unlink(missing_ok=True)
        part.with_name(part.name + ".sha256").unlink(missing_ok=True)
        print("Checksum mismatch; retrying.", file=sys.stderr)
    raise RuntimeError(f"Checksum mismatch after retry: {target}")


def ensure_space(model: dict) -> None:
    missing = sum(spec["size"] for spec in model["assets"].values() if not (LOCAL_ROOT / spec["path"]).is_file())
    if shutil.disk_usage(LOCAL_ROOT).free < missing + 3 * 1024**3:
        raise RuntimeError("Not enough free space for the selected model and runtime.")


def install_model(model: dict) -> dict[str, Path]:
    ensure_space(model)
    return {name: download(name, spec) for name, spec in model["assets"].items()}


def check_model(model: dict) -> bool:
    okay = True
    for name, spec in model["assets"].items():
        path = LOCAL_ROOT / spec["path"]
        valid = verified(path, spec, full=True)
        print(f"{name:12} {'OK' if valid else 'MISSING / INVALID'}  {path}")
        okay &= valid
    return okay


def _extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination)
    children = [path for path in destination.iterdir() if path.name != "__MACOSX"]
    return children[0] if len(children) == 1 and children[0].is_dir() else destination


def _runtime_ready() -> bool:
    try:
        marker = json.loads(RUNTIME_MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return marker == {"comfyui": COMFY_COMMIT, "gguf": GGUF_COMMIT} and (RUNTIME_SOURCE / "main.py").is_file() and (RUNTIME_SOURCE / "custom_nodes" / "ComfyUI-GGUF" / "nodes.py").is_file() and RUNTIME_PYTHON.is_file()


def ensure_runtime(reinstall: bool = False) -> None:
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("Image generation is tested and supported on Apple Silicon macOS.")
    if _runtime_ready() and not reinstall:
        return
    print("Installing ComfyUI + ComfyUI-GGUF (first run only)...")
    archives = {name: download(name, spec) for name, spec in RUNTIME_FILES.items()}
    staging_parent = RUNTIMES_ROOT
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_parent, prefix="comfy-install-") as temporary:
        temp = Path(temporary)
        uv_root = _extract(archives["uv"], temp / "uv")
        comfy_root = _extract(archives["comfyui"], temp / "comfy")
        gguf_root = _extract(archives["gguf"], temp / "gguf")
        if RUNTIME_SOURCE.exists():
            shutil.rmtree(RUNTIME_SOURCE)
        shutil.move(str(comfy_root), str(RUNTIME_SOURCE))
        extension = RUNTIME_SOURCE / "custom_nodes" / "ComfyUI-GGUF"
        extension.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(gguf_root), str(extension))
        UV_BINARY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(uv_root / "uv", UV_BINARY)
        UV_BINARY.chmod(0o755)
    venv = RUNTIME_PYTHON.parents[1]
    if venv.exists():
        shutil.rmtree(venv)
    uv_environment = os.environ.copy()
    uv_environment["UV_PYTHON_INSTALL_DIR"] = str(RUNTIMES_ROOT / "python")
    uv_environment["UV_CACHE_DIR"] = str(CACHE_ROOT / "uv")
    subprocess.run([str(UV_BINARY), "venv", "--managed-python", "--python", "3.13", str(venv)], check=True, env=uv_environment)
    subprocess.run([str(UV_BINARY), "pip", "install", "--python", str(RUNTIME_PYTHON), "-r", str(RUNTIME_SOURCE / "requirements.txt"), "-r", str(RUNTIME_SOURCE / "custom_nodes" / "ComfyUI-GGUF" / "requirements.txt")], check=True, env=uv_environment)
    RUNTIME_MARKER.write_text(json.dumps({"comfyui": COMFY_COMMIT, "gguf": GGUF_COMMIT}) + "\n", encoding="utf-8")


def configure_models(paths: dict[str, Path]) -> None:
    if MODELS_DIRECTORY.exists():
        shutil.rmtree(MODELS_DIRECTORY)
    targets = {"diffusion": MODELS_DIRECTORY / "diffusion_models", "text_encoder": MODELS_DIRECTORY / "text_encoders", "vae": MODELS_DIRECTORY / "vae"}
    for key, directory in targets.items():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / paths[key].name).symlink_to(paths[key])
    for directory in (OUTPUT_DIRECTORY, INPUT_DIRECTORY, TEMP_DIRECTORY, USER_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)


def _workflow_template_path(name: str) -> Path | None:
    path = RUNTIME_PYTHON.parents[1] / "lib" / "python3.13" / "site-packages" / "comfyui_workflow_templates_json" / "templates" / name
    return path if path.is_file() else None


def workflow_nodes(workflow: dict):
    """Yield top-level and subgraph nodes from a ComfyUI workflow."""
    yield from workflow.get("nodes", [])
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def prepared_workflow(model_key: str, model: dict, paths: dict[str, Path]) -> dict | None:
    source = _workflow_template_path(model["template"])
    if source is None:
        return None
    workflow = json.loads(source.read_text(encoding="utf-8"))
    for node in workflow_nodes(workflow):
        node_type = node.get("type")
        if node_type == "UNETLoader":
            if not model.get("native_diffusion"):
                node["type"] = "UnetLoaderGGUF"
                node["widgets_values"] = [paths["diffusion"].name]
                node["widgets_values_named"] = {"unet_name": paths["diffusion"].name}
            else:
                node["widgets_values"] = [paths["diffusion"].name, "default"]
                node["widgets_values_named"] = {"unet_name": paths["diffusion"].name, "weight_dtype": "default"}
        elif node_type == "CLIPLoader":
            clip_type = "flux2" if model_key == "flux2" else "qwen_image"
            if not model.get("native_text_encoder"):
                node["type"] = "CLIPLoaderGGUF"
                node["widgets_values"] = [paths["text_encoder"].name, clip_type]
                node["widgets_values_named"] = {"clip_name": paths["text_encoder"].name, "type": clip_type}
            else:
                node["widgets_values"] = [paths["text_encoder"].name, clip_type, "default"]
                node["widgets_values_named"] = {"clip_name": paths["text_encoder"].name, "type": clip_type, "device": "default"}
        elif node_type == "VAELoader":
            node["widgets_values"] = [paths["vae"].name]
            node["widgets_values_named"] = {"vae_name": paths["vae"].name}
        elif node_type == "Flux2Scheduler" and node.get("widgets_values"):
            node["widgets_values"][0] = model["steps"]
            node.setdefault("widgets_values_named", {})["steps"] = model["steps"]
        elif node_type == "CFGGuider" and node.get("widgets_values"):
            node["widgets_values"][0] = model["cfg"]
            node.setdefault("widgets_values_named", {})["cfg"] = model["cfg"]
        elif node_type == "KSampler" and len(node.get("widgets_values", [])) >= 4:
            node["widgets_values"][2:4] = [model["steps"], model["cfg"]]
            node.setdefault("widgets_values_named", {}).update(
                {"steps": model["steps"], "cfg": model["cfg"]})
        named = node.get("widgets_values_named", {})
        if {"unet_name", "clip_name", "vae_name"}.issubset(named) and len(node.get("widgets_values", [])) >= 12:
            model_names = [paths["diffusion"].name, paths["text_encoder"].name, paths["vae"].name]
            node["widgets_values"][9:12] = model_names
            named.update(unet_name=model_names[0], clip_name=model_names[1], vae_name=model_names[2])
    return workflow


def install_workflow(model_key: str, model: dict, paths: dict[str, Path]) -> Path | None:
    workflow = prepared_workflow(model_key, model, paths)
    if workflow is None:
        return None
    workflows = USER_DIRECTORY / "default" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    destination = workflows / f"Local Model Playground - {model['short_label']}.json"
    destination.write_text(json.dumps(workflow, ensure_ascii=False), encoding="utf-8")
    return destination


def job_workflow(model_key: str, paths: dict[str, Path], **params) -> dict | None:
    """Build the editable ComfyUI graph stored with one submitted job and PNG."""
    workflow = prepared_workflow(model_key, MODELS[model_key], paths)
    if workflow is None:
        return None
    for node in workflow_nodes(workflow):
        node_type = node.get("type")
        values = node.setdefault("widgets_values", [])
        named = node.setdefault("widgets_values_named", {})
        if {"prompt", "steps", "width", "height", "seed"}.issubset(named):
            values[:9] = [params["prompt"], params["negative_prompt"], params["cfg"],
                          params["steps"], params["width"], params["height"],
                          params["sampler"], params["scheduler"], params["seed"]]
            named.update(prompt=params["prompt"], negative_prompt=params["negative_prompt"],
                         cfg=params["cfg"], steps=params["steps"], width=params["width"],
                         height=params["height"], scheduler=params["sampler"],
                         scheduler_1=params["scheduler"], seed=params["seed"])
        elif node_type == "ResolutionSelector" and len(values) >= 3:
            ratio = "1:1 (Square)" if params["width"] == params["height"] else f"{params['width']}:{params['height']}"
            values[:3] = [ratio, params["width"] * params["height"] / 1_000_000, 8]
            named.update(aspect_ratio=ratio, megapixels=values[1], multiple=8)
        if node_type == "TextEncodeQwenImage21" and len(values) >= 3:
            values[:3] = [params["prompt"], params["negative_prompt"], params["width"]]
            named.update(prompt=params["prompt"], negative_prompt=params["negative_prompt"],
                         resolution=params["width"])
        elif node_type == "KSampler" and len(values) >= 6:
            values[:6] = [params["seed"], "fixed", params["steps"], params["cfg"],
                          params["sampler"], params["scheduler"]]
            named.update(seed=params["seed"], control_after_generate="fixed",
                         steps=params["steps"], cfg=params["cfg"],
                         sampler_name=params["sampler"], scheduler=params["scheduler"])
        elif node_type == "CLIPTextEncode" and values:
            text = params["negative_prompt"] if "Negative" in node.get("title", "") else params["prompt"]
            values[0] = text
        elif node_type in ("EmptyFlux2LatentImage", "EmptyLatentImage") and len(values) >= 2:
            values[:2] = [params["width"], params["height"]]
        elif node_type == "RandomNoise" and values:
            values[:2] = [params["seed"], "fixed"]
        elif node_type == "KSamplerSelect" and values:
            values[0] = params["sampler"]
        elif node_type == "Flux2Scheduler" and len(values) >= 3:
            values[:3] = [params["steps"], params["width"], params["height"]]
            named.update(steps=params["steps"], width=params["width"], height=params["height"])
        elif node_type == "CFGGuider" and values:
            values[0] = params["cfg"]
            named["cfg"] = params["cfg"]
    return workflow


def choose_model(default: str = "flux2") -> str:
    choices = list(MODELS.items())
    print("\nChoose an image model:")
    for index, (key, model) in enumerate(choices, 1):
        size = sum(x["size"] for x in model["assets"].values()) / 1e9
        print(f"  {index}. {model['label']} - {size:.1f} GB{' (default)' if key == default else ''}")
    while True:
        answer = input("Select a number or press Enter for the default: ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1][0]
        print(f"Choose a number from 1 to {len(choices)}.")


def port_busy(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def free_port(preferred: int) -> int:
    if not port_busy(preferred):
        return preferred
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def active_server() -> dict | None:
    try:
        state = json.loads(ACTIVE_STATE.read_text(encoding="utf-8"))
        if healthy(int(state["backend_port"])):
            return state
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    return None


def build_command(port: int) -> list[str]:
    return [str(RUNTIME_PYTHON), str(RUNTIME_SOURCE / "main.py"), "--listen", "127.0.0.1", "--port", str(port), "--models-directory", str(MODELS_DIRECTORY), "--user-directory", str(USER_DIRECTORY), "--output-directory", str(OUTPUT_DIRECTORY), "--input-directory", str(INPUT_DIRECTORY), "--temp-directory", str(TEMP_DIRECTORY), "--disable-auto-launch", "--disable-manager-ui", "--preview-method", "auto"]


def start_backend(command: list[str], log_path: Path) -> subprocess.Popen:
    environment = os.environ.copy()
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    process = subprocess.Popen(command, cwd=RUNTIME_SOURCE, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1, start_new_session=(os.name != "nt"), env=environment)
    log = log_path.open("w", encoding="utf-8")

    def relay() -> None:
        assert process.stdout is not None
        with process.stdout, log:
            for line in process.stdout:
                with LOG_LOCK:
                    log.write(line)
                    log.flush()
                    print(f"[ComfyUI] {line}", end="", flush=True)

    process.log_thread = threading.Thread(target=relay, daemon=True)  # type: ignore[attr-defined]
    process.log_thread.start()  # type: ignore[attr-defined]
    return process


def stop_backend(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if process is not None and hasattr(process, "log_thread"):
        process.log_thread.join(timeout=5)  # type: ignore[attr-defined]


def api_workflow(model_key: str, paths: dict[str, Path], prompt: str, width: int,
                 height: int, steps: int, seed: int, cfg: float | None = None,
                 negative_prompt: str = "", sampler: str | None = None,
                 scheduler: str | None = None, preset: str | None = None) -> dict:
    del preset
    cfg = MODELS[model_key]["cfg"] if cfg is None else cfg
    sampler = sampler or "euler"
    scheduler = scheduler or ("flux2" if model_key == "flux2" else "simple")
    base = {
        "1": {"class_type": "UNETLoader" if model_key == "qwen21" else "UnetLoaderGGUF", "inputs": {"unet_name": paths["diffusion"].name, **({"weight_dtype": "default"} if model_key == "qwen21" else {})}},
        "2": {"class_type": "CLIPLoader" if model_key == "qwen21" else "CLIPLoaderGGUF", "inputs": {"clip_name": paths["text_encoder"].name, "type": "flux2" if model_key == "flux2" else "qwen_image", **({"device": "default"} if model_key == "qwen21" else {})}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": paths["vae"].name}},
    }
    if model_key == "flux2":
        base.update({
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 0]}},
            "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
            "9": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
            "10": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": cfg}},
            "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["6", 0]}},
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
            "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "benchmark/flux2"}},
        })
    else:
        if width != height:
            raise RuntimeError("Qwen-Image-2.1 benchmark requires a square size.")
        base.update({
            "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": negative_prompt, "vae": ["3", 0], "resolution": width}},
            "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["4", 2], "denoise": 1.0}},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["3", 0]}},
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "benchmark/qwen21"}},
        })
    return base


def _json_request(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def run_benchmark(port: int, workflow: dict, timeout: int = 3600) -> tuple[float, Path]:
    started = time.monotonic()
    prompt_id = _json_request(f"http://127.0.0.1:{port}/prompt", {"prompt": workflow})["prompt_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        item = _json_request(f"http://127.0.0.1:{port}/history/{prompt_id}").get(prompt_id)
        if item:
            status = item.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError(f"ComfyUI workflow failed: {status}")
            for output in item.get("outputs", {}).values():
                images = output.get("images", [])
                if images:
                    image = images[0]
                    return time.monotonic() - started, OUTPUT_DIRECTORY / image.get("subfolder", "") / image["filename"]
        time.sleep(1)
    raise RuntimeError(f"Benchmark timed out after {timeout} seconds.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reinstall", action="store_true", help="reinstall the image runtime")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--comfyui", action="store_true", help="open the raw ComfyUI interface")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--benchmark", action="store_true", help="generate one timed image and exit")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--report", help="write benchmark JSON here")
    args = parser.parse_args(argv)

    interactive = sys.stdin.isatty() and not args.no_prompt
    model_key = args.model or (choose_model() if interactive else "flux2")
    model = MODELS[model_key]
    if args.check:
        runtime_ok = _runtime_ready()
        print(f"runtime      {'OK' if runtime_ok else 'MISSING'}  ComfyUI + ComfyUI-GGUF")
        return 0 if runtime_ok and check_model(model) else 1
    running = active_server()
    if running:
        url = f"http://127.0.0.1:{running['port']}"
        if running.get("model") == model_key:
            print(f"{model['short_label']} is already running at {url}")
            if not args.no_browser:
                webbrowser.open(url, new=2)
            return 0
        print(f"Another image model ({running.get('model')}) is already running at {url}.", file=sys.stderr)
        return 1

    process = None
    try:
        ensure_runtime(args.reinstall)
        paths = install_model(model)
        configure_models(paths)
        workflow_path = install_workflow(model_key, model, paths)
        if args.download_only:
            print(f"{model['short_label']} and its runtime are installed and verified.")
            return 0
        ui_port = free_port(args.port)
        backend_port = free_port(ui_port + 1)
        logs = LOGS_ROOT / "image"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{model_key}.log"
        process = start_backend(build_command(backend_port), log_path)
        ACTIVE_STATE.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_STATE.write_text(json.dumps({"pid": os.getpid(), "port": ui_port,
                                            "backend_port": backend_port, "model": model_key}), encoding="utf-8")
        for name in ("SIGHUP", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        started = time.monotonic()
        while time.monotonic() - started < 180:
            if process.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with code {process.returncode}. See {log_path}")
            if healthy(backend_port):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"ComfyUI startup timed out. See {log_path}")
        startup_seconds = time.monotonic() - started
        comfy_url = f"http://127.0.0.1:{backend_port}"
        print(f"ComfyUI backend ready in {startup_seconds:.1f}s: {comfy_url}")
        if args.benchmark:
            if args.runs < 1:
                parser.error("--runs must be at least 1")
            steps = args.steps or model["steps"]
            prompt = "A small red fox sitting in a sunlit meadow, detailed natural photograph"
            results = []
            for index in range(args.runs):
                run_seed = args.seed + index
                graph = api_workflow(model_key, paths, prompt, args.width, args.height, steps, run_seed)
                seconds, image = run_benchmark(backend_port, graph)
                results.append({"run": index + 1, "seed": run_seed,
                                "seconds": round(seconds, 2), "output": str(image)})
                print(f"Benchmark {index + 1}/{args.runs}: {model_key} {args.width}x{args.height}, {steps} steps, {seconds:.1f}s")
            report = {
                "schema": 1, "timestamp": datetime.now(timezone.utc).isoformat(),
                "platform": {"system": platform.system(), "machine": platform.machine()},
                "kind": "image", "model": model_key, "width": args.width, "height": args.height,
                "steps": steps, "cfg": model["cfg"], "seed": args.seed,
                "startup_seconds": round(startup_seconds, 2), "runs": results, "passed": True,
            }
            destination = Path(args.report).expanduser() if args.report else (
                LOGS_ROOT / "benchmarks" / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{model_key}.json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Benchmark report: {destination}")
            return 0
        from playground.image.webui import ImageUIState, start_server
        ui_state = ImageUIState(backend_port, comfy_url, model_key, model, paths,
                                OUTPUT_DIRECTORY, api_workflow,
                                HISTORY_ROOT / "image-history.json", job_workflow)
        ui_server, _ui_thread = start_server(ui_port, ui_state)
        url = comfy_url if args.comfyui else f"http://127.0.0.1:{ui_port}"
        print(f"Playground UI: http://127.0.0.1:{ui_port}")
        if not args.no_browser:
            webbrowser.open(url, new=2)
        print(f"Prepared workflow: {workflow_path}" if workflow_path else "Use the built-in workflow templates.")
        print("Press Ctrl+C to stop and release the model.")
        while process.poll() is None:
            time.sleep(0.5)
        raise RuntimeError(f"ComfyUI exited unexpectedly with code {process.returncode}. See {log_path}")
    except KeyboardInterrupt:
        print("\nStopping ComfyUI...")
    except (OSError, RuntimeError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        if "ui_server" in locals():
            ui_server.shutdown()
            ui_server.server_close()
        stop_backend(process)
        try:
            state = json.loads(ACTIVE_STATE.read_text(encoding="utf-8"))
            if state.get("pid") == os.getpid():
                ACTIVE_STATE.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
