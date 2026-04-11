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

## 9. sitecustomize.py Startup Hook

The project includes a `sitecustomize.py` at the repo root that is auto-imported by Python on startup (when running from the repo directory). It performs two patches:

1. **OpenAI SDK type fix** — adds `ChatCompletionFunctionToolParam` if missing from the installed SDK version (compatibility shim for different openai package versions)
2. **SGLANG_DISABLE_AMX workaround** — when `SGLANG_DISABLE_AMX=1`, monkey-patches SGLang's AMX fast-paths to avoid segfaults on certain CPU/NUMA configurations

```bash
# If you hit segfaults in AMX attention on specific NUMA setups:
export SGLANG_DISABLE_AMX=1
```

## 10. libiomp5 Discovery Chain

The SGLang server script searches for `libiomp5.so` in this order:

1. `SGLANG_LIB_IOMP` (explicit override)
2. Active Python env: `{PY_PREFIX}/lib/libiomp5.so`
3. System: `/usr/lib/x86_64-linux-gnu/libiomp5.so`
4. `ldconfig -p` lookup
5. Other conda envs under the same conda root

If not found, continues without it (warn). Set `SGLANG_REQUIRE_IOMP=1` to fail hard.

## Checklist

- [ ] AMX ISA enabled
- [ ] BF16 dtype
- [ ] tcmalloc + tbbmalloc + libiomp5 preloaded
- [ ] Torch compile enabled with matching batch size
- [ ] MALLOC_ARENA_MAX=1
- [ ] NUMA pinning configured
- [ ] Overlap schedule disabled
- [ ] sitecustomize.py present (OpenAI SDK + AMX workaround)

## Related

- [AMX](../concepts/amx.md) | [Torch Compile](../concepts/torch-compile.md) | [KV Cache](../concepts/kv-cache.md)
- [Multi-Instance Guide](multi-instance.md)
- [SGLang Backend](../entities/backends/sglang.md)
- [Environment Variables Reference](environment-variables.md)
- [Profiling & Tracing](profiling-and-tracing.md)
