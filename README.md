# Local Model Playground

A dependency-light playground for trying local text and image models one at a time behind one interactive command.

This is an experimental workstation tool, not a production frontend or multi-model server. Starting one model at a time keeps unified-memory use predictable and makes switching models explicit.

## Quick start

Python 3.9 or newer is required. The launchers use only the Python standard library.

```bash
python3 launch.py
```

The interactive picker offers three text profiles and two image profiles. The selected model is downloaded and verified on first use, its local service opens in a browser, and `Ctrl+C` releases the model and runtime.

Direct commands are also available:

```bash
python3 launch.py llm --model compact
python3 launch.py llm --model uncensored
python3 launch.py image --model flux2
python3 launch.py image --model qwen21
```

Only one launcher session may run at a time. Text and image backends are deliberately separate processes because they use different runtimes and can individually consume most of a Mac's unified memory.

## Models

| Kind | Launch value | Download | Notes |
| --- | --- | ---: | --- |
| Text | `compact` | 5.95 GB | PTQ1_0 base plus refusal-ablation LoRA; lowest memory |
| Text | `uncensored` | 13.50 GB | Q3_K_M; Hugging Face approval required once |
| Text | `direct` | 17.00 GB | Q4_K_M; highest memory use |
| Image | `flux2` | 5.3 GB | FLUX.2 Klein Base 4B Q4; recommended image model |
| Image | `qwen21` | 14.2 GB | Official Qwen-Image-2.1 INT8/W4A8 ComfyUI weights; quality experiments |

Balanced defaults are selected per model rather than globally:

| Model | Default | Why |
| --- | --- | --- |
| `compact` | 16K context, Q8 KV cache | useful working window at about 7 GiB measured resident memory |
| `uncensored` | 16K context, Q8 KV cache | leaves headroom around the 13.5 GB weights |
| `direct` | 8K context, Q8 KV cache | protects a 24 GB machine from memory pressure around the 17 GB weights |
| `flux2` | 12 steps, CFG 4 | measured quality/latency balance for the non-distilled Base checkpoint |
| `qwen21` | 25 steps, CFG 1 | matches the official ComfyUI template baseline |

Explicit command-line options and `LOCAL_LLM_CTX` still override these defaults.

Text inference uses a pinned PrismML llama.cpp runtime and retains the original WebUI. Its local proxy adds rolling conversation compression. Image inference uses pinned ComfyUI and ComfyUI-GGUF releases with Metal acceleration behind a small playground UI.

The image UI opens in a simple mode with Fast, Balanced, and Detail presets, common sizes, seed control, parameter advice, and full local history. Expand **Advanced parameters** to edit steps, CFG, sampler, scheduler, dimensions, and the negative prompt; the UI explains the effect and recommended range of each choice. During generation it shows the current stage, step progress, ETA, and live diffusion preview. Results include their absolute local path. History supports parameter reuse plus individual, selected, or complete deletion; deleting a history item also removes its generated image. Each run is a native ComfyUI Job and embeds its matching editable workflow in the Job record and PNG. Use **Open ComfyUI** (or launch with `--comfyui`) when you deliberately want the full node graph.

## Local storage

All large or generated files are centralized under `.local/` and excluded from Git:

```text
.local/
├── assets/           everything fetched or installed; safe to rebuild
│   ├── models/
│   ├── runtimes/
│   └── downloads/
├── history/          every user-side artifact: images, inputs, chat memory,
│                     ComfyUI data, logs, and benchmarks
└── work/             non-user temporary state and caches
```

Put the cache on another disk without changing the repository:

```bash
MODEL_PLAYGROUND_HOME=/Volumes/Models/playground python3 launch.py
```

Inspect and clean it with:

```bash
python3 launch.py storage
python3 launch.py clean                 # clear all user data plus temporary work data
python3 launch.py clean --runtimes      # also remove rebuildable runtimes; asks first
python3 launch.py clean --models        # also remove model weights; asks first
python3 launch.py clean --all           # remove all local assets; asks first
```

Invalid downloads are quarantined with an `.invalid-*` suffix and removed by the default cleanup. Model weights are never duplicated as backups by this project.

## Verification

Check the Python/toolchain, installed components, executable runtimes, and free space:

```bash
python3 launch.py doctor
python3 launch.py doctor --full         # additionally hash every installed model
python3 -m unittest discover -s tests -v
python3 -m compileall -q launch.py playground tests
```

Run a real isolated text-model inference check with:

```bash
python3 launch.py verify --model compact
python3 launch.py verify --model compact --context 32768
```

Run a repeatable baseline benchmark. Missing runtimes and weights are installed automatically,
then a JSON report is written under `.local/logs/benchmarks/`:

```bash
python3 launch.py benchmark --model compact
python3 launch.py benchmark --model flux2
python3 launch.py benchmark --model flux2 --steps 20 --runs 2
```

The text benchmark records startup, prompt/generation throughput, wall time, token usage, and
resident memory. The image benchmark uses reproducible sequential seeds and records cold/warm
generation time plus output paths. See [`docs/benchmark-report.md`](docs/benchmark-report.md)
for the measured baseline and interpretation.

The image launcher bootstraps uv, Python 3.13, ComfyUI, ComfyUI-GGUF, the lightweight UI, and the selected model automatically. No separate frontend or backend installation is needed. It can verify an installed model and runtime without starting them:

```bash
python3 launch.py image --model flux2 --check
```

## Platform notes

- The text launcher selects pinned runtime packages for macOS arm64/x64, Linux arm64/x64, and Windows x64. Only Apple Silicon has been exercised by this project.
- The image launcher currently supports Apple Silicon macOS. It only needs the system Python used to enter the launcher; its image runtime and dependencies are installed in `.local/` automatically.
- Services bind to `127.0.0.1` and have no authentication. Do not expose them directly to a LAN or the internet.

Exact model revisions, sizes, hashes, and upstream sources are kept in [`playground/llm/assets.lock.json`](playground/llm/assets.lock.json) and [`playground/image/launcher.py`](playground/image/launcher.py). Repository source code uses the MIT License in `LICENSE`.
