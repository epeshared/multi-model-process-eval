---
title: Auto-Test Framework
created: 2026-04-11
updated: 2026-04-11
tags: [guide, auto-test, benchmark, sweep, emon]
sources: [scripts/auto-test/embedding/run_auto_test.py, scripts/auto-test/embedding/config_fix_token_len.json]
---

# Auto-Test Framework

Automated multi-config benchmark orchestration with server lifecycle management, CPU affinity, [EMON](../concepts/emon.md) integration, and CSV reporting.

## Overview

`scripts/auto-test/embedding/run_auto_test.py` runs a sequence of jobs defined in a JSON config. For each job it:

1. Starts backend server(s) with specified environment and CPU affinity
2. Waits for server readiness (polling health endpoint)
3. Runs the benchmark script with per-job parameters
4. Optionally collects EMON energy data during the run
5. Stops servers (if configured)
6. Aggregates results across repeats (mean ± std)

## CLI

```bash
cd scripts/auto-test/embedding
python run_auto_test.py --config config_fix_token_len.json [options]
```

| Flag | Description |
|------|-------------|
| `--config` | Config JSON path (required) |
| `--only <name>` | Run only named jobs (repeatable) |
| `--skip <name>` | Skip named jobs (repeatable) |
| `--dry-run` | Print commands without executing |
| `--tee` | Stream logs to terminal and file |
| `--restart-servers` | Force server restart between jobs |
| `--stop-servers-after-job` | Stop servers after each job |
| `--reparse-run-id <id>` | Re-parse old logs for a given run ID |

## Config Schema

```json
{
  "version": 1,
  "description": "Embedding benchmark sweep",
  "script_aliases": {
    "embed": "scripts/embedding/run_fix_token_len.sh"
  },
  "defaults": {
    "cwd": ".",
    "result_dir": "scripts/auto-test/embedding/result",
    "timeout_sec": 600,
    "restart_servers": false,
    "stop_server_after_job": false,
    "env": {
      "DTYPE": "bfloat16",
      "DEVICE": "cpu"
    }
  },
  "servers": {
    "sglang": {
      "enabled": true,
      "start_script": "scripts/embedding/sglang/start_sglang_server.sh",
      "cwd": "scripts/embedding/sglang",
      "args": [],
      "env": {
        "MODEL_DIR": "/path/to/model",
        "BATCH_SIZE": "16",
        "PORT": "30000"
      },
      "numactl": {
        "cpunodebind": "0",
        "membind": "0",
        "physcpubind": "0-23,48-71"
      },
      "env_from_job": ["MODEL_DIR", "BATCH_SIZE"],
      "ready": {
        "url": "http://127.0.0.1:30000/health",
        "timeout_sec": 300,
        "poll_interval_sec": 5
      }
    }
  },
  "jobs": [
    {
      "name": "qwen3-emb-0.6b-sglang-tok64-bs16",
      "script": "embed",
      "args": [],
      "warmup_runs": 1,
      "emon_enable": false,
      "env": {
        "MODEL": "Qwen3-Embedding-0.6B",
        "BACKEND": "sglang",
        "BATCH_SIZE": "16",
        "SYNTHETIC_TOKEN_LEN": "64",
        "MAX_SAMPLES": "10000"
      }
    }
  ]
}
```

### Key Config Fields

| Section | Field | Description |
|---------|-------|-------------|
| `defaults` | `timeout_sec` | Per-job timeout |
| `defaults` | `restart_servers` | Restart servers between jobs |
| `servers.<name>` | `start_script` | Path to server startup script |
| `servers.<name>` | `numactl` | CPU/memory NUMA binding |
| `servers.<name>` | `env_from_job` | Inherit these env vars from the job |
| `servers.<name>` | `ready.url` | Health endpoint to poll |
| `servers.<name>` | `ready.timeout_sec` | Max wait for server readiness |
| `jobs[*]` | `emon_enable` | Enable Intel [EMON](../concepts/emon.md) collection |
| `jobs[*]` | `warmup_runs` | Number of warmup runs before timed runs |

## CPU Affinity

Server processes are pinned via `numactl`:

```json
"numactl": {
  "cpunodebind": "0",
  "membind": "0",
  "physcpubind": "0-23,48-71"
}
```

This translates to:
```bash
numactl --cpunodebind=0 --membind=0 --physcpubind=0-23,48-71 bash start_sglang_server.sh
```

The `AUTO_TEST_CPU_EXPR` environment variable can also override at the job level.

## EMON Integration

When `emon_enable: true`, the framework:

1. Spawns `emon -collect-edp` as a subprocess before the benchmark
2. Lets it collect CPU performance counters during the run
3. Stops EMON after the benchmark completes
4. Post-processes via `pyedp` to generate `.xlsx` summary

See [EMON](../concepts/emon.md) for details.

## Output

Results are written to `{result_dir}/{run_id}/`:

```
result/{run_id}/
├── run.log              # Full orchestration log
├── aggregate.csv        # Merged results from all jobs
├── {job_name}/
│   ├── stdout.log       # Job output
│   ├── metrics.json     # Parsed metrics
│   └── emon/            # EMON data (if enabled)
│       ├── edp_output/
│       └── summary.xlsx
```

## Related

- [Running Benchmarks](running-benchmarks.md) — manual benchmark workflow
- [Multi-Instance Guide](multi-instance.md) — NUMA and port assignment
- [EMON](../concepts/emon.md) — Intel energy monitoring
- [Remote Deployment](remote-deployment.md) — scale-test framework (builds on auto-test)
- [Agent Skills Reference](agent-skills-reference.md) — `auto_test` skill
