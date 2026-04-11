---
title: Intel EMON (Energy Monitoring)
created: 2026-04-11
updated: 2026-04-11
tags: [concept, emon, energy, profiling, intel]
sources: [scripts/auto-test/embedding/run_auto_test.py, requirements-emon.txt]
---

# Intel EMON (Energy Monitoring)

EMON (Event Monitor) is Intel's CPU performance counter and energy data collection tool, part of the SEP (Sampling Enabling Product) suite. This project integrates EMON for power-aware benchmarking.

## What It Measures

| Category | Example Metrics |
|----------|----------------|
| Power | Package power (watts), DRAM power |
| Frequency | CPU operating frequency, uncore frequency, DDR data rate |
| Cache | L1D MPI, L2 MPI, LLC MPI (misses per instruction) |
| Memory | Read/write bandwidth (MB/sec), NUMA local vs remote % |
| TMA | Frontend Bound, Backend Bound, Core Bound, Memory Bound (%) |
| Utilization | CPU utilization %, CPI, core IPC, C-state residency |

## How It Works

### Collection

The [Auto-Test Framework](../guides/auto-test-framework.md) spawns EMON during benchmark execution:

```
emon -collect-edp /opt/intel/sep/config/edp/pyedp_config.txt
```

EMON runs as a subprocess alongside the benchmark, collecting hardware performance counters via Intel PMU (Performance Monitoring Unit).

### Post-Processing

After collection, PyEDP processes the raw data:

```bash
pip install -r requirements-emon.txt  # installs pyedp
```

PyEDP generates:
- `summary.xlsx` — per-socket and system-level metrics
- Socket view: per-socket breakdown of all TMA/cache/memory metrics
- System view: aggregated system-wide metrics

### Analysis Integration

The `scale_analyze` skill extracts key metrics from EMON `.xlsx` files into CSV:

- `emon_socket_metrics.csv` — per-socket view (one row per socket × metric)
- System-view metrics included in `summary_pivot.csv`

## Config

### Auto-Test Config

```json
{
  "jobs": [{
    "name": "benchmark-with-emon",
    "emon_enable": true,
    ...
  }]
}
```

### Scale-Test Config

```json
{
  "emon": {
    "emon_enable": true,
    "args": ["emon", "-process-pyedp", "/opt/intel/sep/config/edp/pyedp_config.txt"]
  }
}
```

## Key Metrics for CPU Inference

| Metric | Why It Matters |
|--------|---------------|
| Package power (watts) | TPS-per-watt efficiency comparison |
| Memory bandwidth (MB/sec) | Memory-bound detection for large models |
| TMA Backend_Bound (%) | Identifies bottleneck type (core vs memory) |
| NUMA remote reads (%) | Detects cross-socket memory access penalties |
| LLC MPI | Cache pressure indicator |
| Core C6 residency (%) | Thread sleep efficiency |

## Prerequisites

- Intel SEP installed (`/opt/intel/sep/`)
- Root or `perf_event_paranoid` permissions
- `pip install -r requirements-emon.txt`

## Related

- [Auto-Test Framework](../guides/auto-test-framework.md) — EMON integration in benchmarks
- [CPU Optimization Guide](../guides/cpu-optimization.md) — hardware tuning
- [AMX](amx.md) — Intel matrix acceleration
