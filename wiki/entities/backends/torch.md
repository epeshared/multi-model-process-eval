---
title: Torch Backend
created: 2026-04-10
updated: 2026-04-10
tags: [backend, torch, transformers, ipex]
sources: []
---

# Torch Backend

Local inference using PyTorch + HuggingFace Transformers, with optional IPEX/AMX acceleration.

## Features

- Direct model loading via `AutoModel` / `AutoModelForVision2Seq`
- CLIP support via CLIPModel
- Mean pooling for text embeddings
- IPEX/AMX acceleration for CPU (see [AMX](../../concepts/amx.md))
- Dtype support: `bfloat16`, `float16`, `float32`
- Attention implementation configurable (`attn_implementation`)

## Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `DEVICE` | cpu | Device to use |
| `DTYPE` | bfloat16 | Model precision |
| `USE_AMX` | TRUE | Enable IPEX/AMX |
| `trust_remote_code` | True | Allow custom model code |

## CPU Optimization

When `USE_AMX=TRUE` on CPU:
- Sets `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`
- Enables IPEX optimizations
- See [CPU Optimization Guide](../../guides/cpu-optimization.md)

## Supported Tasks

| Task | Status |
|------|--------|
| [Embedding](../tasks/embedding.md) | ✅ |
| [VL](../tasks/vl.md) | ✅ |
| [Qwen3 LLM](../tasks/qwen3-llm.md) | — |
| [Omni](../tasks/omni.md) | — |

## Related

- [SGLang Backend](sglang.md) — server-based alternative
- [vLLM Backend](vllm.md) — server-based alternative
- [AMX](../../concepts/amx.md) | [Torch Compile](../../concepts/torch-compile.md)
