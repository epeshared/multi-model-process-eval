---
title: Environment Variables Reference
created: 2026-04-11
updated: 2026-04-11
tags: [guide, environment, configuration, reference]
sources: [scripts/, src/agent_service/config.py, sitecustomize.py]
---

# Environment Variables Reference

Complete reference for all environment variables used across the project.

## Model & Data

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | — | Logical model key (e.g. `Qwen3-Embedding-0.6B`) |
| `MODEL_ID` | — | HuggingFace model ID or served model name |
| `MODEL_DIR` | — | Local path to model weights |
| `SERVED_MODEL_NAME` | — | Model name registered on vLLM server (required for vllm-http) |
| `DATASET` | varies | Data source: `yahoo`, `flickr8k`, `synthetic`, `custom` |
| `MAX_SAMPLES` | 10000 | Max input samples |

## Backend & Server

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND` | `torch` | Backend: `torch`, `sglang`, `sglang-offline`, `vllm`, `vllm-http` |
| `DEVICE` | `cpu` | Device: `cpu` or `cuda` |
| `BASE_URL` | `http://127.0.0.1:30000` | HTTP server URL |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `30000` | Server port |

## Benchmark Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | 16 | Batch size (also sets `--torch-compile-max-bs`) |
| `WARMUP_SAMPLES` | 1 | Samples excluded from timing |
| `DTYPE` | `bfloat16` | Model precision: `bfloat16`, `float16`, `float32` |
| `MODE` | — | Synthetic data mode: `input_len` or `token_len` |
| `SYNTHETIC_INPUT_LEN` | — | Fixed character length for synthetic input |
| `SYNTHETIC_TOKEN_LEN` | — | Fixed token count for synthetic input |

## SGLang Server

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_USE_CPU_ENGINE` | `1` | Force CPU engine |
| `SGLANG_MAX_TOTAL_TOKENS` | — | KV cache token capacity |
| `SGLANG_CONTEXT_LENGTH` | — | Max context length per request |
| `SGLANG_MEM_FRACTION_STATIC` | — | Static memory fraction (0.0–1.0) |
| `SGLANG_CHUNKED_PREFILL_SIZE` | — | Chunked prefill token limit |
| `SGLANG_MAX_PREFILL_TOKENS` | — | Max prefill tokens |
| `SGLANG_DISABLE_RADIX_CACHE` | — | Disable prefix cache (`1`/`true`) |
| `SGLANG_ENABLE_MULTIMODAL` | `0` | Enable multimodal support (`1` to enable) |
| `SGLANG_NUMA_NODE` | — | Explicit NUMA node hint |
| `SGLANG_TORCH_PROFILER_DIR` | — | Server-side profiler output directory |
| `SGLANG_DISABLE_AMX` | — | Disable AMX fast-paths (`1`/`true`; via sitecustomize.py) |

## SGLang Python Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_PYTHON` | — | Override Python interpreter path |
| `SGLANG_CONDA_ENV` | — | Conda env name for `conda run` |
| `SGLANG_REQUIRE_IOMP` | `0` | Fail if libiomp5 not found (`1` to enforce) |
| `SGLANG_LIB_IOMP` | — | Explicit libiomp5.so path override |

## Intel CPU Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `DNNL_MAX_CPU_ISA` | `AVX512_CORE_AMX` | Max ISA for OneDNN ([AMX](../concepts/amx.md)) |
| `DNNL_VERBOSE` | `0` | OneDNN verbose logging |
| `IPEX_DISABLE_AUTOCAST` | `1` | Disable IPEX autocast (avoids uint64 copy_kernel bug) |
| `MALLOC_ARENA_MAX` | `1` | Limit glibc malloc arenas (reduce memory fragmentation) |
| `USE_AMX` | — | Enable AMX in task scripts |
| `C10_DISABLE_NUMA` | — | Disable PyTorch NUMA logic (`1` to disable) |

## Profiling

| Variable | Default | Description |
|----------|---------|-------------|
| `PROFILE` | `0` | Enable torch profiler (`1`) |
| `PROFILE_ACTIVITIES` | `CPU` | Profiler activities: `CPU`, `CUDA` |
| `PROFILE_RECORD_SHAPES` | `0` | Record tensor shapes |
| `PROFILE_OUT_DIR` | `./profile_traces` | Trace output directory |

See [Profiling & Tracing Guide](profiling-and-tracing.md).

## Auto-Test

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_TEST_CPU_EXPR` | — | CPU affinity expression for numactl |

See [Auto-Test Framework](auto-test-framework.md).

## Agent Service

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_OPENAI_BASE_URL` | `http://127.0.0.1:8000/v1` | LLM backend URL |
| `AGENT_OPENAI_API_KEY` | `""` | LLM API key |
| `AGENT_MODEL` | `""` (required) | Model name for tool-call loop |
| `AGENT_MAX_TOOL_STEPS` | `5` (0–20) | Max tool calls per chat turn |

See [Agent Skills Reference](agent-skills-reference.md).

## VL-Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_DIR` | — | Image directory path |
| `IMAGE_SIZE` | — | Filter tag in filename (e.g. `512x512`) |
| `IMAGE_TRANSPORT` | `data-url` | `data-url`, `base64`, `path`, `url` |
| `EMBEDDING_HTTP_TIMEOUT` | `900` | HTTP timeout (seconds) |

See [VL-Embedding Task](../entities/tasks/vl-embedding.md).

## HuggingFace

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace mirror URL (for remote bootstrap) |
| `HUGGINGFACE_HUB_CACHE` | `~/.cache/huggingface` | Model cache directory |

## Related

- [CPU Optimization Guide](cpu-optimization.md) — Intel-specific tuning
- [Running Benchmarks](running-benchmarks.md) — benchmark workflow
- [SGLang Backend](../entities/backends/sglang.md) — server configuration
- [vLLM Backend](../entities/backends/vllm.md) — server configuration
