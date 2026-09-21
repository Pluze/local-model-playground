# Model Defaults and Benchmark Report

Date: 2026-09-21

Test host: MacBook Air, Apple M5 (10 CPU cores), 24 GB unified memory, macOS 27.0.
All tests used the pinned runtimes and model revisions in this repository. Services were bound
to localhost and only one model was loaded at a time.

## Recommended defaults

| Model | Context / sampling | Recommendation |
| --- | --- | --- |
| Compact text | 16K, one slot, Q8 KV, temperature 1.0, top-p 0.95, top-k 20 | Default text model; lowest download and measured memory use |
| Uncensored text | 16K, one slot, Q8 KV, temperature 1.0, top-p 0.95, top-k 20 | Better quantization quality when 14+ GB of model memory is acceptable |
| Direct text | 8K, one slot, Q8 KV, temperature 1.0, top-p 0.95, top-k 20 | Shorter window preserves headroom for the 17 GB Q4 model on 24 GB machines |
| FLUX.2 Base 4B | 12 steps, CFG 4 | Best measured interactive balance; increase to 20 for a final render |
| Qwen-Image-2.1 | 25 steps, CFG 1 | Official ComfyUI template baseline; use mainly for quality experiments |

The text sampling values follow the model-family generation guidance for thinking mode. Clients
may still send per-request sampling values. Q8 KV halves the cache element width relative to F16;
the measured comparison below showed lower memory and no deterministic-output change.

## Measurements

### All text profiles

Each profile completed a warm-up and a real deterministic generation with its recommended context
and Q8 KV cache. The generation prompt was identical across models.

| Model | Startup | Output | Generation throughput | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Compact | 2.03 s | 125 tokens | 12.82 tok/s | 10.88 s |
| Uncensored | 9.08 s | 115 tokens | 7.66 tok/s | 15.87 s |
| Direct | 8.06 s | 113 tokens | 6.37 tok/s | 18.48 s |

All three returned non-empty, normally terminated answers. macOS process RSS is retained in the
machine-readable reports, but it does not consistently include memory-mapped model pages or Metal
allocations and therefore should not be used to compare total unified-memory demand.

### Compact text: Q8 versus F16 KV

The benchmark used a 16K context, one warm-up, then two deterministic 125-token responses.

| KV type | Startup | Generation throughput | Wall time | Resident memory |
| --- | ---: | ---: | ---: | ---: |
| Q8 | 3.05 s | 12.60 / 12.14 tok/s | 11.13 / 10.71 s | 6,995 / 7,150 MiB |
| F16 | 2.04 s | 12.47 / 11.80 tok/s | 11.19 / 11.04 s | 7,399 / 7,559 MiB |

Q8 saved about 407 MiB on average and was about 2% faster in this short run. Both modes produced
the same deterministic 692-character answer. Startup varies with filesystem cache and should not
be used to claim a Q8 startup advantage.

The earlier 32K capacity test remains valid: the compact profile accepted a 24,531-token prompt,
but that test took 400.15 seconds. A 16K daily default avoids reserving a large window that most
interactive sessions do not use; 32K remains available through `--context 32768`.

### FLUX.2 Base 4B

The same 512x512 fox prompt and seed were rendered at 12 and 20 steps with CFG 4.

| Steps | Measured time | Observation |
| ---: | ---: | --- |
| 12 | 92.88 s cold, 97.78 s second seed | crisp subject, coherent grass/fur detail |
| 20 | 140.37 s cold | same-seed result was visually almost indistinguishable at 512x512 |

A later full default regression run after the UI work completed in 61.43 seconds at 12 steps. An
end-to-end request through the new playground UI completed in 44.23 seconds at the 8-step Fast
preset; its history entry and image-serving route were then read back successfully.

The fanless machine slowed from roughly 6.0 to 8.2 seconds per step during consecutive runs, so
reports keep every run rather than hiding thermal behavior in one average. Twelve steps reduces
first-image latency by about one third without an obvious loss on the comparison prompt.

### Qwen-Image-2.1

The complete 14.2 GB profile was checksum-verified and rendered the 512x512 comparison prompt at
the recommended 25 Euler/simple steps and CFG 1 in 184.28 seconds. The output was sharp and
coherent. This validates the full default rather than extrapolating from the earlier 2- and 8-step
smoke tests. The reference Diffusers pipeline describes 40 steps, but on this fanless host the
25-step ComfyUI baseline is the better default; 12 and 40 steps remain explicit Fast and Detail
choices.

All five model profiles were physically present, fully hash-verified, loaded one at a time, and
used for real inference in this validation run.

## Image UI design

The default interface intentionally borrows the approachable preset-first pattern from Fooocus and
Easy Diffusion, keeps recent outputs and parameter reuse close to the canvas as Invoke does, and
retains an escape hatch to raw ComfyUI similar to SwarmUI's advanced workflow path. ComfyUI remains
the pinned execution engine; the project adds no second frontend framework or plugin dependency.

## Re-running the baseline

```bash
python3 launch.py benchmark --model compact
python3 launch.py benchmark --model uncensored
python3 launch.py benchmark --model direct
python3 launch.py benchmark --model flux2
python3 launch.py benchmark --model qwen21
```

Each command installs missing assets, runs the selected model in isolation, stops it, and writes a
machine-readable JSON report to `.local/logs/benchmarks/`. Use `--context`, `--steps`, `--runs`,
`--width`, `--height`, and `--report` for controlled comparisons. Image runs start from the same
base seed and increment it per run so ComfyUI cannot return a cached graph result.

Primary references:

- [Ternary Bonsai generation parameters](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- [llama.cpp server context and KV-cache options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [Official Qwen-Image-2.1 ComfyUI workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json)
- [Qwen-Image-2.1 reference pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)
- [Fooocus preset-first UI](https://github.com/lllyasviel/Fooocus/blob/main/webui.py)
- [Easy Diffusion UI overview](https://github.com/easydiffusion/easydiffusion/wiki/UI-Overview)
- [Invoke output history and canvas](https://github.com/invoke-ai/InvokeAI)
- [SwarmUI advanced workflow path](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Advanced%20Usage.md)
