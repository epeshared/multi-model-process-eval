---
title: Omni Task
created: 2026-04-10
updated: 2026-04-10
tags: [task, multimodal, omni]
sources: []
---

# Omni Task

Multimodal (image + text, future: audio) benchmarks using Qwen2.5-Omni models.

## Entry Points

| File | Purpose |
|------|---------|
| `src/tasks/omni.py` | Task logic + backend dispatch |
| `scripts/omni/run_omni.py` | CLI entry point |
| `scripts/omni/run_qwen_omni_synthetic.sh` | Synthetic benchmark |

## Supported Backends

- [SGLang](../backends/sglang.md) HTTP
- [vLLM](../backends/vllm.md) offline + HTTP

Backend implementations in `src/tasks/omni_backends/`.

## Notes

- Currently image-dominant in benchmark scripts
- Uses OpenAI-compatible multimodal chat API for HTTP backends
- Shares the session pattern with other tasks

## Related

- [Qwen2.5-Omni Model](../models/qwen25-omni.md)
- [VL Task](vl.md) — vision-language sibling task
