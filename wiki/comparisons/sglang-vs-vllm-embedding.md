---
title: SGLang vs vLLM — Embedding
created: 2026-04-10
updated: 2026-04-10
tags: [comparison, sglang, vllm, embedding]
sources: []
---

# SGLang vs vLLM — Embedding

Comparison of SGLang and vLLM for embedding workloads.

## Feature Comparison

| Feature | SGLang | vLLM |
|---------|--------|------|
| Native embedding endpoint | `/encode` | — |
| OpenAI-compat endpoint | `/v1/embeddings` | `/v1/embeddings` |
| Chat-style embedding | — | `EmbeddingChatRequest` |
| [Torch Compile](../concepts/torch-compile.md) | ✅ (server flag) | ✅ (engine config) |
| Radix cache | ✅ (can disable) | — |
| Profiling endpoints | `/start_profile`, `/stop_profile` | — |
| Encoding formats | float | float, base64 |
| Image embedding | ✅ (with `--enable-multimodal`) | ✅ |
| Tokenizer batch encode | ✅ (`--enable-tokenizer-batch-encode`) | — |

## CPU Server Startup

| Aspect | SGLang | vLLM |
|--------|--------|------|
| Script | `scripts/*/sglang/start_sglang_server.sh` | `scripts/*/vllm/start_vllm_server.sh` |
| LD_PRELOAD | tcmalloc + tbbmalloc + libiomp5 | varies |
| AMX attention | `--attention-backend intel_amx` | — |
| Torch compile flag | `--enable-torch-compile` | engine-level |

## Performance Notes

_No benchmark results ingested yet. Add results to `raw/results/` and run an ingest to populate._

## When to Use Which

- **SGLang**: preferred for CPU workloads with AMX, native profiling, radix cache
- **vLLM**: preferred for GPU workloads, OpenAI-compatible ecosystem, base64 encoding

## Related

- [SGLang Backend](../entities/backends/sglang.md)
- [vLLM Backend](../entities/backends/vllm.md)
- [Backend Feature Matrix](backend-feature-matrix.md)
