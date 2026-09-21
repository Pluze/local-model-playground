# Local LLM validation record

Platform: Apple Silicon macOS. Model: Ternary-Bonsai-2-27B-PTQ1_0 GGUF with the OrcaBonsai LoRA. Runtime: PrismML `prism-b10709-9a9394a`.

The isolated `python3 launch.py verify --context 32768` run on 2026-09-19 reported:

| Check | Result |
| --- | --- |
| Pinned model and adapter SHA-256 | Passed |
| Short inference | Passed |
| Streamed completion ID | Present |
| `reasoning_end` control API | `success: true`; final content followed |
| Configured context | 32,768 tokens |
| Long prompt | 24,531 prompt tokens in the completion response (75% of window) |
| Long prompt processing and short generation | 400.15 seconds |
| Process cleanup | Passed; no model process remained |

The long request deliberately capped output at 16 tokens, so it ended with `finish_reason: length`. This check establishes that the configured window accepted a prompt at 75% occupancy. It does not establish answer quality or reliable recall across all 24K tokens. This result is machine-specific.

The original WebUI's Skip reasoning button was observed failing in one long browser conversation with `no completion id for the active message`. The same runtime's control API succeeded in an isolated stream, so the remaining issue concerns that WebUI message state rather than missing server support. The launcher uses the upstream WebUI unchanged.
