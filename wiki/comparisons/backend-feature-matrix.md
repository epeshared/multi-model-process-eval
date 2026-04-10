---
title: Backend Feature Matrix
created: 2026-04-10
updated: 2026-04-10
tags: [comparison, backend, feature-matrix]
---

# Backend Feature Matrix

Complete feature support across all backends and tasks.

## Task × Backend Support

| Task | torch | sglang HTTP | sglang offline | vllm HTTP | vllm offline |
|------|:-----:|:-----------:|:--------------:|:---------:|:------------:|
| Embedding | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen3 LLM | — | ✅ | — | ✅ | — |
| VL | ✅ | ✅ | ✅ | ✅ | ✅ |
| Omni | — | ✅ | — | ✅ | ✅ |

## Feature × Backend

| Feature | torch | sglang HTTP | sglang offline | vllm HTTP | vllm offline |
|---------|:-----:|:-----------:|:--------------:|:---------:|:------------:|
| CPU inference | ✅ | ✅ | ✅ | ✅ | ✅ |
| CUDA inference | ✅ | ✅ | ✅ | ✅ | ✅ |
| [AMX](../concepts/amx.md) acceleration | ✅ | ✅ | ✅ | — | — |
| [Torch Compile](../concepts/torch-compile.md) | — | ✅ | ✅ | — | — |
| Streaming | — | ✅ | — | ✅ | — |
| Profiling | code-level | endpoints | code-level | — | — |
| Image input | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tensor parallelism | — | ✅ | ✅ | ✅ | ✅ |
| Data parallelism | — | — | ✅ | — | — |

## Model × Backend (Tested)

| Model | torch | sglang | vllm |
|-------|:-----:|:------:|:----:|
| Qwen3-Embedding-0.6B | ✅ | ✅ | ✅ |
| Qwen3-Embedding-4B | ✅ | ✅ | ✅ |
| CLIP ViT-B/32 | ✅ | ✅ | ✅ |
| CLIP ViT-L/14-336 | ✅ | ✅ | ✅ |
| Youtu-Embedding-FP16 | ✅ | ✅ | ✅ |
| Qwen3-0.6B | — | ✅ | ✅ |
| Qwen3-1.7B | — | ✅ | ✅ |
| Qwen3-4B | — | ✅ | ✅ |
| Qwen2.5-VL-3B | ✅ | ✅ | ✅ |
| Qwen2.5-VL-7B | ✅ | ✅ | ✅ |
| Qwen2.5-Omni-3B | — | ✅ | ✅ |
| Qwen2.5-Omni-7B | — | ✅ | ✅ |

## Related

- [SGLang vs vLLM — Embedding](sglang-vs-vllm-embedding.md)
- [Adding a New Backend](../guides/adding-backend.md)
