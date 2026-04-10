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

| Variable | Purpose |
|----------|---------|
| `PORT` | Listen port |
| `MODEL_DIR` | Model path |
| `tp_size` | Tensor parallelism degree |
| `gpu_memory_utilization` | GPU memory fraction |
| `max_model_len` | Maximum sequence length |

## Supported Tasks

| Task | HTTP | Offline |
|------|:----:|:-------:|
| [Embedding](../tasks/embedding.md) | ✅ | ✅ |
| [Qwen3 LLM](../tasks/qwen3-llm.md) | ✅ | — |
| [VL](../tasks/vl.md) | ✅ | ✅ |
| [Omni](../tasks/omni.md) | ✅ | ✅ |

## Notes

- vLLM HTTP supports multimodal chat via `EmbeddingChatRequest`
- Streaming in vLLM-http measures [TTFT](../../concepts/ttft.md) and [TPOT](../../concepts/tpot.md) per-token

## Related

- [SGLang Backend](sglang.md) — alternative serving backend
- [Torch Backend](torch.md) — local transformers inference
- [SGLang vs vLLM — Embedding](../../comparisons/sglang-vs-vllm-embedding.md)
