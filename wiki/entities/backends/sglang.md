---
title: SGLang Backend
created: 2026-04-10
updated: 2026-04-10
tags: [backend, sglang, server]
sources: []
---

# SGLang Backend

SGLang provides both an HTTP server mode and an offline engine mode for local inference.

## Modes

### SGLang HTTP (`sglang`)

An HTTP server with OpenAI-compatible API endpoints:

- `/encode` — native SGLang embedding endpoint
- `/v1/embeddings` — OpenAI-compatible embedding
- `/v1/chat/completions` — chat completion
- `/start_profile`, `/stop_profile` — profiling control

**Server startup**: see `scripts/*/sglang/start_sglang_server.sh`

### SGLang Offline (`sglang-offline`)

Direct `sglang.Engine` usage for local inference without HTTP overhead:

- Embedding mode with tensor/data parallelism
- Optional [torch compile](../../concepts/torch-compile.md)

## CPU Server Configuration

Key flags in `start_sglang_server.sh`:

```bash
--device cpu
--dtype bfloat16
--attention-backend intel_amx        # AMX acceleration
--enable-torch-compile               # torch.compile
--torch-compile-max-bs $BATCH_SIZE   # max compiled batch size
--is-embedding                       # embedding mode
--skip-server-warmup
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_DIR` | youtu-embedding-fp16 | Model path |
| `BATCH_SIZE` | 16 | Max batch size for torch compile |
| `HOST` | 0.0.0.0 | Listen host |
| `PORT` | 30000 | Listen port |
| `SGLANG_MAX_TOTAL_TOKENS` | — | KV cache token cap |
| `SGLANG_CONTEXT_LENGTH` | — | Max context length |
| `SGLANG_MEM_FRACTION_STATIC` | — | Static memory fraction |
| `SGLANG_NUMA_NODE` | — | NUMA node hint |
| `SGLANG_ENABLE_MULTIMODAL` | 0 | Enable multimodal support |
| `SGLANG_DISABLE_RADIX_CACHE` | — | Disable radix cache |

## LD_PRELOAD Libraries

The server script preloads for performance:
- `libtcmalloc.so.4` — Google's thread-caching malloc
- `libtbbmalloc.so.2` — Intel TBB scalable allocator
- `libiomp5.so` — Intel OpenMP runtime (auto-discovered, optional)

## Qwen3 LLM Server Differences

The Qwen3 LLM server (`scripts/qwen3/sglang/start_sglang_server.sh`) differs from the embedding server:

- No `--is-embedding` flag (generative mode)
- `--device ${DEVICE}` (supports both `cpu` and `cuda`)
- `--disable-cuda-graph --disable-piecewise-cuda-graph` on CPU
- `--max-total-tokens` default 65536 (larger context window)
- No `--attention-backend intel_amx` or `--enable-tokenizer-batch-encode`

## Server Scripts

| Task | Script | Key Flags |
|------|--------|-----------|
| Embedding | `scripts/embedding/sglang/start_sglang_server.sh` | `--is-embedding --attention-backend intel_amx` |
| Qwen3 LLM | `scripts/qwen3/sglang/start_sglang_server.sh` | `--disable-cuda-graph` |
| VL | `scripts/vl/sglang/start_sglang_server.sh` | `--enable-multimodal` |
| VL (CUDA) | `scripts/vl/sglang/start_sglang_server_cuda.sh` | `--device cuda` |
| Omni | `scripts/omni/sglang/start_sglang_server.sh` | `--enable-multimodal` |
| Omni (CUDA) | `scripts/omni/sglang/start_sglang_server_cuda.sh` | `--device cuda` |

## Profiling Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/start_profile` | POST | Start torch profiler (output to `SGLANG_TORCH_PROFILER_DIR`) |
| `/stop_profile` | POST | Stop profiler and flush traces |
| `/health` | GET | Health check |

See [Profiling & Tracing](../../guides/profiling-and-tracing.md) for usage.

## Supported Tasks

| Task | HTTP | Offline |
|------|:----:|:-------:|
| [Embedding](../tasks/embedding.md) | ✅ | ✅ |
| [Qwen3 LLM](../tasks/qwen3-llm.md) | ✅ | — |
| [VL](../tasks/vl.md) | ✅ | ✅ |
| [VL-Embedding](../tasks/vl-embedding.md) | ✅ | — |
| [Omni](../tasks/omni.md) | ✅ | — |

## Related

- [vLLM Backend](vllm.md) — alternative serving backend
- [CPU Optimization Guide](../../guides/cpu-optimization.md) — LD_PRELOAD, AMX
- [Environment Variables](../../guides/environment-variables.md) — SGLANG_* vars
- [Torch Backend](torch.md) — local transformers inference
- [CPU Optimization Guide](../../guides/cpu-optimization.md)
- [AMX](../../concepts/amx.md)
