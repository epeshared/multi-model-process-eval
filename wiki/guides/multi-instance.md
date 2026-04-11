---
title: Multi-Instance Guide
created: 2026-04-10
updated: 2026-04-10
tags: [guide, multi-instance, numa, scaling]
---

# Multi-Instance Guide

Running multiple server instances on the same machine for scale testing.

## Why Multi-Instance

- Simulate production deployments with N workers
- Test CPU affinity and NUMA-local performance
- Measure scaling efficiency (throughput vs. instance count)

## Memory Caps (Critical)

Without explicit caps, each SGLang instance tries to allocate KV cache based on total system memory, causing massive over-allocation when multiple start simultaneously.

**Always set:**

```bash
export SGLANG_MAX_TOTAL_TOKENS=4096
export SGLANG_CONTEXT_LENGTH=2048
export SGLANG_MEM_FRACTION_STATIC=0.3
```

See [KV Cache](../concepts/kv-cache.md) for details.

## CPU Affinity

The auto-test framework (`scripts/auto-test/`) supports CPU affinity via:

```bash
export AUTO_TEST_CPU_EXPR="0-23,48-71"  # NUMA node 0 cores
```

This is passed to `numactl` or `taskset` to pin each instance to specific cores.

## Port Assignment

Each instance needs a unique port:

```bash
PORT=30000 ./start_sglang_server.sh &
PORT=30001 ./start_sglang_server.sh &
PORT=30002 ./start_sglang_server.sh &
```

## Auto-Test Framework

`scripts/auto-test/embedding/run_auto_test.py` automates multi-instance testing:

- JSON config defining instance count, parameters, and sweep ranges
- Automatic server start/stop between jobs
- CPU affinity assignment per instance
- Emon (Intel energy monitoring) integration
- Multi-repeat aggregation (mean ± std)
- CSV output

Example:

```bash
cd scripts/auto-test/embedding
python run_auto_test.py --config config_fix_token_len.json
```

## Scale Test Scripts

`scripts/scale-test/` contains pre-built configurations for:
- Embedding scaling tests
- Qwen3 scaling tests
- VL-Embedding scaling tests

## Related

- [KV Cache](../concepts/kv-cache.md)
- [CPU Optimization Guide](cpu-optimization.md)
- [Remote Deployment Guide](remote-deployment.md) — multi-host SSH deployment and remote testing
- [Batch Size Tuning](../concepts/batch-size-tuning.md)
- [Batch Size Tuning](../concepts/batch-size-tuning.md)
