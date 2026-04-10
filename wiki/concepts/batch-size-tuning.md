---
title: Batch Size Tuning
created: 2026-04-10
updated: 2026-04-10
tags: [concept, performance, tuning]
---

# Batch Size Tuning

Batch size significantly affects throughput and latency trade-offs in all tasks.

## Two Batch Sizes

There are two distinct batch size concepts in this project:

### Client Batch Size

The number of samples sent per API call from the benchmark script.

- Controlled by `BATCH_SIZE` in `run_embedding.py` etc.
- Default: 100 for embedding tasks
- Affects client-side batching and throughput measurement

### Server Compiled Batch Size

The max batch size that [torch compile](torch-compile.md) optimizes for.

- Controlled by `BATCH_SIZE` (or legacy `SERVER_BATCH_SIZE`) in server scripts
- Default: 16
- Passed as `--torch-compile-max-bs`

## Impact

| Batch Size | Throughput | Latency per Sample | Memory |
|-----------|------------|-------------------|--------|
| Small (1–4) | Low | Low | Minimal |
| Medium (8–32) | High | Moderate | Moderate |
| Large (64+) | Plateau/decrease | High | High |

## Tips

- Match server compiled BS to the actual client BS for best performance
- For multi-instance runs, reduce per-instance BS to avoid memory contention
- Use the auto-test framework to sweep batch sizes systematically

## Related

- [Torch Compile](torch-compile.md)
- [KV Cache](kv-cache.md)
- [Multi-Instance Guide](../guides/multi-instance.md)
