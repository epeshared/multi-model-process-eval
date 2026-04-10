---
title: CPU Optimization Guide
created: 2026-04-10
updated: 2026-04-10
tags: [guide, cpu, optimization, intel, amx]
---

# CPU Optimization Guide

Best practices for maximizing inference throughput on Intel Xeon CPUs.

## 1. Enable AMX

```bash
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

In SGLang: `--attention-backend intel_amx`

In torch backend: `USE_AMX=TRUE`

See [AMX](../concepts/amx.md) for details.

## 2. Use BFloat16

```bash
--dtype bfloat16
```

AMX tiles natively operate on BF16. Using FP32 misses hardware acceleration.

## 3. Preload Performance Libraries

The SGLang server scripts preload three libraries via `LD_PRELOAD`:

| Library | Purpose |
|---------|---------|
| `libtcmalloc.so.4` | Google's thread-caching malloc — reduces allocation overhead |
| `libtbbmalloc.so.2` | Intel TBB scalable allocator — NUMA-aware allocation |
| `libiomp5.so` | Intel OpenMP runtime — optimized threading |

The script auto-discovers `libiomp5` from: active Python env → system locations → ldconfig → other conda envs. Override with `SGLANG_LIB_IOMP=/path/to/libiomp5.so`.

## 4. Enable Torch Compile

```bash
--enable-torch-compile
--torch-compile-max-bs 16
```

Captures and optimizes the computation graph. See [Torch Compile](../concepts/torch-compile.md).

## 5. Memory Allocator Tuning

```bash
export MALLOC_ARENA_MAX=1    # Reduce glibc arena count
```

Prevents excessive virtual memory usage with many threads.

## 6. IPEX Configuration

```bash
export IPEX_DISABLE_AUTOCAST=1   # Avoid uint64 copy_kernel issues
export DNNL_VERBOSE=0            # Disable OneDNN logging
```

## 7. NUMA Awareness

For multi-socket systems:

- Use `numactl --cpunodebind=N --membind=N` to pin to a NUMA node
- Pass `SGLANG_NUMA_NODE=N` to hint the server
- Consider `C10_DISABLE_NUMA=1` if PyTorch NUMA detection causes issues

## 8. Disable Overlap Schedule

```bash
--disable-overlap-schedule
```

On CPU, overlapping decode/prefill can hurt throughput. Disable for predictable serial execution.

## Checklist

- [ ] AMX ISA enabled
- [ ] BF16 dtype
- [ ] tcmalloc + tbbmalloc + libiomp5 preloaded
- [ ] Torch compile enabled with matching batch size
- [ ] MALLOC_ARENA_MAX=1
- [ ] NUMA pinning configured
- [ ] Overlap schedule disabled

## Related

- [AMX](../concepts/amx.md) | [Torch Compile](../concepts/torch-compile.md) | [KV Cache](../concepts/kv-cache.md)
- [Multi-Instance Guide](multi-instance.md)
- [SGLang Backend](../entities/backends/sglang.md)
