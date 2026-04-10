---
title: AMX — Intel Advanced Matrix Extensions
created: 2026-04-10
updated: 2026-04-10
tags: [concept, hardware, intel, cpu, optimization]
---

# AMX — Intel Advanced Matrix Extensions

Hardware acceleration for matrix operations on Intel Xeon CPUs (Sapphire Rapids and later).

## How It's Used

This project is primarily optimized for AMX-enabled CPUs:

### OneDNN / IPEX Configuration

```bash
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX   # Enable AMX instruction set
export IPEX_DISABLE_AUTOCAST=1              # Avoid uint64 copy_kernel issues
```

### SGLang Server

```bash
--attention-backend intel_amx    # Use AMX for attention computation
--dtype bfloat16                 # AMX natively supports BF16
```

### Torch Backend

```bash
USE_AMX=TRUE    # Enables IPEX/AMX optimizations
```

## Performance Impact

- Significant speedup for GEMM (matrix multiplication) operations
- Best with `bfloat16` dtype — native AMX tile format
- Combined with [torch compile](torch-compile.md) for additional graph-level optimization

## Prerequisites

- Intel Xeon with AMX support (4th Gen Xeon Scalable / Sapphire Rapids or later)
- Linux kernel 5.16+ (for AMX tile support)
- OneDNN / IPEX libraries

## Related

- [Torch Compile](torch-compile.md) — complementary optimization
- [CPU Optimization Guide](../guides/cpu-optimization.md)
- [SGLang Backend](../entities/backends/sglang.md)
