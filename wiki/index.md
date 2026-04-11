---
title: Wiki Index
created: 2026-04-10
updated: 2026-04-10
tags: [meta]
---

# Wiki Index

Content catalog for the multi-model-process-eval knowledge base.

## Overview

| Page | Summary | Updated |
|------|---------|---------|
| [overview.md](overview.md) | High-level project architecture and purpose | 2026-04-10 |

## Entities — Models

| Page | Summary | Updated |
|------|---------|---------|
| [Qwen3-Embedding](entities/models/qwen3-embedding.md) | Qwen3-Embedding family (0.6B, 4B) — text embedding model | 2026-04-10 |
| [CLIP](entities/models/clip.md) | OpenAI CLIP (ViT-B/32, ViT-L/14-336) — multimodal embedding | 2026-04-10 |
| [Youtu-Embedding](entities/models/youtu-embedding.md) | Tencent Youtu embedding model (FP16) | 2026-04-10 |
| [Qwen3-LLM](entities/models/qwen3-llm.md) | Qwen3 LLM family (0.6B, 1.7B, 4B) — text generation | 2026-04-10 |
| [Qwen2.5-VL](entities/models/qwen25-vl.md) | Qwen2.5-VL (3B, 7B) — vision-language model | 2026-04-10 |
| [Qwen2.5-Omni](entities/models/qwen25-omni.md) | Qwen2.5-Omni (3B, 7B) — multimodal (image + audio) | 2026-04-10 |

## Entities — Backends

| Page | Summary | Updated |
|------|---------|---------|
| [SGLang](entities/backends/sglang.md) | SGLang backend — HTTP server + offline engine | 2026-04-10 |
| [vLLM](entities/backends/vllm.md) | vLLM backend — HTTP server + offline engine | 2026-04-10 |
| [Torch](entities/backends/torch.md) | PyTorch/Transformers local backend with IPEX/AMX | 2026-04-10 |

## Entities — Tasks

| Page | Summary | Updated |
|------|---------|---------|
| [Embedding](entities/tasks/embedding.md) | Text and image embedding task | 2026-04-10 |
| [Qwen3 LLM](entities/tasks/qwen3-llm.md) | LLM chat / text generation task | 2026-04-10 |
| [VL](entities/tasks/vl.md) | Vision-language chat task | 2026-04-10 |
| [Omni](entities/tasks/omni.md) | Multimodal (image + audio) chat task | 2026-04-10 |

## Concepts

| Page | Summary | Updated |
|------|---------|---------|
| [TTFT](concepts/ttft.md) | Time To First Token — first-token latency metric | 2026-04-10 |
| [TPOT](concepts/tpot.md) | Time Per Output Token — generation throughput metric | 2026-04-10 |
| [AMX](concepts/amx.md) | Intel Advanced Matrix Extensions for CPU inference | 2026-04-10 |
| [Torch Compile](concepts/torch-compile.md) | PyTorch 2.x torch.compile for inference acceleration | 2026-04-10 |
| [KV Cache](concepts/kv-cache.md) | Key-Value cache management for transformer inference | 2026-04-10 |
| [Batch Size Tuning](concepts/batch-size-tuning.md) | Batch size impact on throughput and latency | 2026-04-10 |

## Guides

| Page | Summary | Updated |
|------|---------|---------|
| [CPU Optimization](guides/cpu-optimization.md) | Best practices for Intel CPU inference (AMX, IPEX, LD_PRELOAD) | 2026-04-10 |
| [Multi-Instance](guides/multi-instance.md) | Running multiple server instances with CPU affinity and memory caps | 2026-04-10 |
| [Adding a New Backend](guides/adding-backend.md) | How to add a new backend implementation | 2026-04-10 |
| [Running Benchmarks](guides/running-benchmarks.md) | End-to-end benchmark workflow | 2026-04-10 |
| [Remote Deployment](guides/remote-deployment.md) | Multi-host SSH deployment, bootstrap, and remote testing | 2026-04-11 |

## Comparisons

| Page | Summary | Updated |
|------|---------|---------|
| [SGLang vs vLLM — Embedding](comparisons/sglang-vs-vllm-embedding.md) | Embedding throughput comparison across backends | 2026-04-10 |
| [Backend Feature Matrix](comparisons/backend-feature-matrix.md) | Feature support across all backends and tasks | 2026-04-10 |

## Sources

_No sources ingested yet. Add raw documents to `raw/sources/` and run an ingest operation._
