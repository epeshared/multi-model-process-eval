---
title: Torch Compile
created: 2026-04-10
updated: 2026-04-10
tags: [concept, optimization, pytorch]
---

# Torch Compile

PyTorch 2.x `torch.compile` for graph-capture and JIT optimization of inference workloads.

## Usage in This Project

### SGLang Server

```bash
--enable-torch-compile              # Enable compilation
--torch-compile-max-bs $BATCH_SIZE  # Max batch size to compile for
```

The `BATCH_SIZE` env var (default: 16) determines the compiled batch size. Inputs exceeding this are processed without compilation.

### SGLang Offline

Set `enable_torch_compile=True` and `torch_compile_max_bs=N` in engine config.

## How It Works

1. First inference triggers graph capture and compilation (slow)
2. Subsequent inferences reuse the compiled graph (fast)
3. Different batch sizes may trigger recompilation

## Trade-offs

| Aspect | Impact |
|--------|--------|
| First-run latency | Significantly increased (compilation overhead) |
| Steady-state throughput | Improved (optimized graph execution) |
| Memory | Slightly increased (compiled graph storage) |
| Batch size flexibility | Reduced (compiled for fixed max BS) |

## Tips

- Use `--skip-server-warmup` and handle warmup separately in benchmark scripts
- Match compiled batch size to your actual workload batch size
- Combined with [AMX](amx.md) for maximum CPU throughput

## Related

- [AMX](amx.md) — hardware acceleration companion
- [Batch Size Tuning](batch-size-tuning.md)
- [SGLang Backend](../entities/backends/sglang.md)
