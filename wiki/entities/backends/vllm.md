---
title: vLLM Backend
created: 2026-04-10
updated: 2026-04-10
tags: [backend, vllm, server]
sources: []
---

# vLLM Backend

vLLM provides OpenAI-compatible HTTP serving and offline engine modes.

## Modes

### vLLM HTTP (`vllm-http`)

OpenAI-compatible API endpoints:

- `/v1/embeddings` — embedding endpoint
- `/v1/chat/completions` — chat completion (streaming supported)
- Encoding formats: `base64` or native float arrays

**Server startup**: see `scripts/*/vllm/start_vllm_server.sh`

### vLLM Offline (`vllm`)

Direct vLLM engine with `task="embed"`:

- Tensor parallelism via `tp_size`
- GPU memory utilization tuning
- `max_model_len` control

## Key Configuration

| Variable | Default | Purpose |
|----------|---------|--------|
| `PORT` | 9090 | Listen port |
| `MODEL_DIR` | — | Model path |
| `SERVED_MODEL_NAME` | — | **Required** for vllm-http: model name in API requests |
| `tp_size` | 1 | Tensor parallelism degree |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction |
| `max_model_len` | — | Maximum sequence length |
| `DTYPE` | `bfloat16` | Model precision |
| `CUDA_VISIBLE_DEVICES` | — | Set to `""` to force CPU |

## Server Scripts

| Task | Script | Notes |
|------|--------|-------|
| Embedding (CPU) | `scripts/embedding/vllm/start_vllm_server.sh` | `--runner pooling`, `CUDA_VISIBLE_DEVICES=""` |
| Embedding (CUDA) | `scripts/embedding/vllm/start_vllm_server_cuda.sh` | CUDA build |
| Qwen3 LLM | `scripts/qwen3/vllm/start_vllm_server.sh` | Chat completions |
| VL | `scripts/vl/vllm/start_vllm_server.sh` | Multimodal |
| Omni | `scripts/omni/vllm/start_vllm_server.sh` | Multimodal |

### CPU Server Specifics

- Preloads tcmalloc + tbbmalloc (same as SGLang)
- Preflight checks: vLLM import, FP8 weight detection (fails on CPU)
- Forces `--dtype bfloat16` if float16 requested on CPU
- Uses `--enforce-eager 1` (no torch compilation on CPU)

## Supported Tasks

| Task | HTTP | Offline |
|------|:----:|:-------:|
| [Embedding](../tasks/embedding.md) | ✅ | ✅ |
| [Qwen3 LLM](../tasks/qwen3-llm.md) | ✅ | — |
| [VL](../tasks/vl.md) | ✅ | ✅ |
| [VL-Embedding](../tasks/vl-embedding.md) | ✅ | — |
| [Omni](../tasks/omni.md) | ✅ | ✅ |

## Notes

- `SERVED_MODEL_NAME` must match what the client sends as `model` in API requests
- vLLM HTTP supports multimodal chat via `EmbeddingChatRequest`
- Streaming in vLLM-http measures [TTFT](../../concepts/ttft.md) and [TPOT](../../concepts/tpot.md) per-token
- The `/v1/embeddings` endpoint returns OpenAI-compatible format: `{"data": [{"embedding": [...], "index": 0}], "usage": {...}}`

## Related

- [SGLang Backend](sglang.md) — alternative serving backend
- [Torch Backend](torch.md) — local transformers inference
- [SGLang vs vLLM — Embedding](../../comparisons/sglang-vs-vllm-embedding.md)
- [Environment Variables](../../guides/environment-variables.md)
