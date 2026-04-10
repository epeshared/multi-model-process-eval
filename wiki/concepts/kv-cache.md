---
title: KV Cache
created: 2026-04-10
updated: 2026-04-10
tags: [concept, memory, inference]
---

# KV Cache

Key-Value cache stores intermediate attention states to avoid recomputation during autoregressive generation.

## Configuration in This Project

### SGLang Server

| Variable | Flag | Purpose |
|----------|------|---------|
| `SGLANG_MAX_TOTAL_TOKENS` | `--max-total-tokens` | Total KV cache token budget |
| `SGLANG_CONTEXT_LENGTH` | `--context-length` | Max context per request |
| `SGLANG_MEM_FRACTION_STATIC` | `--mem-fraction-static` | Static memory fraction |
| `SGLANG_DISABLE_RADIX_CACHE` | `--disable-radix-cache` | Disable prefix sharing |

### vLLM

- `max_model_len` — max sequence length
- `gpu_memory_utilization` — GPU memory fraction for KV cache

## Multi-Instance Considerations

When running multiple server instances on the same machine:

> If unset, sglang sizes KV cache based on perceived available memory, which can massively over-allocate when multiple servers start in parallel.

Always set explicit memory caps for [multi-instance](../guides/multi-instance.md) deployments.

## Radix Cache (SGLang)

SGLang's radix cache enables prefix sharing — common prompt prefixes are cached once and shared across requests. Disable with `SGLANG_DISABLE_RADIX_CACHE=1` if not beneficial for your workload (e.g., all-unique prompts in embedding).

## Related

- [SGLang Backend](../entities/backends/sglang.md)
- [Multi-Instance Guide](../guides/multi-instance.md)
- [Batch Size Tuning](batch-size-tuning.md)
