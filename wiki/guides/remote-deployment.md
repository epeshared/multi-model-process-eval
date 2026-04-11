---
title: Remote Deployment & Multi-Host Testing
created: 2026-04-11
updated: 2026-04-11
tags: [guide, remote, ssh, scale-test, deployment, multi-host]
sources: [scripts/scale-test/embedding/run_scale_fix_token_len.py, scripts/scale-test/embedding/conf-remote-example.json, scripts/scale-test/pre-requirements/]
---

# Remote Deployment & Multi-Host Testing

The scale-test framework supports **fully automated remote multi-host deployment and benchmarking** via SSH. A single local command can bootstrap environments on N remote machines, deploy code, execute sweeps in parallel, collect results, and produce aggregated analysis — all with resume support.

## Architecture Overview

```
Local (dispatcher)                    Remote hosts (workers)
┌──────────────────────┐   SSH/SCP   ┌────────────────────────┐
│ run_scale_fix_        │────────────▶│ Worker 1 (10.0.0.11)   │
│   token_len.py        │            │  ├─ pre-requirements    │
│                       │   SSH/SCP  │  ├─ conda env setup     │
│ Reads config JSON     │────────────▶│  ├─ pip install         │
│ Dispatches to hosts   │            │  └─ run sweep locally   │
│ Collects results      │            └────────────────────────┘
│ Merges aggregate.csv  │   SSH/SCP   ┌────────────────────────┐
│                       │────────────▶│ Worker 2 (10.0.0.12)   │
└──────────────────────┘  rsync-back  │  └─ (same flow)        │
                                      └────────────────────────┘
```

## Configuration

Remote deployment uses a **split config** pattern: one JSON for the sweep job, one for remote server definitions. They are merged at runtime.

### Remote Config Structure

```json
{
  "version": 1,
  "description": "Multi-host remote config",
  "run": {
    "servers": [
      {
        "ip": "10.0.0.11",
        "username": "ubuntu",
        "port": 22,
        "identity_file": "~/.ssh/id_rsa",
        "ssh_options": [
          "ProxyCommand=nc -x proxy.example.com:1080 -X 5 %h %p",
          "ConnectTimeout=10",
          "ServerAliveInterval=30"
        ],
        "conda_env": "sglang-cpu",
        "remote_repo_dir": "/home/ubuntu/multi-model-process-eval",
        "remote_result_root": "/home/ubuntu/results",
        "install_requirements": true,
        "requirements_profile": "cpu",
        "pre_requirements": [
          {"file": "scripts/scale-test/pre-requirements/install_miniforge3_linux_x86_64.sh", "stage": "before_conda", "mode": "bash"},
          {"file": "scripts/scale-test/pre-requirements/install_sglang_system_deps.sh", "stage": "before_conda", "mode": "bash"},
          {"file": "scripts/scale-test/pre-requirements/enable_miniforge_conda.sh", "stage": "before_conda", "mode": "source"},
          {"file": "scripts/scale-test/pre-requirements/ensure_conda_env.sh", "stage": "before_conda", "mode": "bash"},
          {"file": "scripts/scale-test/pre-requirements/install_sglang.sh", "stage": "after_conda", "mode": "bash"},
          {"file": "scripts/scale-test/pre-requirements/download-models.sh", "stage": "after_conda", "mode": "bash"}
        ]
      }
    ],
    "remote_repo_dir": "/home/ubuntu/multi-model-process-eval",
    "remote_result_root": "/home/ubuntu/results",
    "ssh": {
      "user": "ubuntu",
      "port": 22,
      "identity_file": "",
      "options": [],
      "password_file": "scripts/scale-test/embedding/passwords.json"
    }
  }
}
```

### Server Entry Fields

| Field | Type | Description |
|-------|------|-------------|
| `ip` / `host` | string | Remote hostname or IP |
| `username` | string | SSH login user |
| `password` | string | SSH password (prefer `identity_file`; if null, reads from `password_file`) |
| `port` | int | SSH port (default: 22) |
| `identity_file` | string | Path to SSH private key |
| `ssh_options` | string[] | Extra SSH `-o` options (ProxyCommand, ConnectTimeout, etc.) |
| `conda_env` | string | Conda environment name on remote |
| `remote_repo_dir` | string | Repo clone path on remote |
| `remote_result_root` | string | Result output path on remote |
| `remote_python` | string/list | Python command override (e.g. `["conda", "run", "-n", "env", "python"]`) |
| `install_requirements` | bool | Whether to `pip install` requirements on remote |
| `requirements_profile` | string | `"cpu"` or `"cuda"` — selects requirements file |
| `pre_requirements` | object[] | Bootstrap scripts to run before the sweep |
| `pre_setup_cmds` | string[] | Arbitrary bash commands before setup |

### Password File Format

```json
{
  "hosts": {
    "10.0.0.11": { "ubuntu": "password-for-ubuntu-on-this-host" },
    "10.0.0.12": { "root": "password" }
  },
  "users": {
    "ubuntu": "fallback-password-for-any-host"
  }
}
```

Resolution order: `hosts.<host>.<user>` → `hosts.<host>` (string) → `users.<user>`.

Password auth uses `sshpass`. Prefer key-based auth (`identity_file`) for production use.

## Execution Flow

### Phase 1: Preflight (Optional)

Before dispatching, verify hosts are reachable:

```bash
# Via agent skill
POST /v1/skills/remote_preflight_fix_token_len
{"config_path": "scripts/scale-test/embedding/config/cloud/my-remote.json"}
```

Checks per host: `whoami`, `hostname`, conda availability, `remote_repo_dir` exists, `remote_result_root` exists.

### Phase 2: Pre-Requirements Bootstrap

Scripts are staged to remote via SCP, then executed in order:

| Stage | Script | Purpose |
|-------|--------|---------|
| `before_conda` | `install_miniforge3_linux_x86_64.sh` | Install Miniforge (conda) from GitHub releases |
| `before_conda` | `install_sglang_system_deps.sh` | System packages: tcmalloc, tbbmalloc, numactl, build-essential |
| `before_conda` | `enable_miniforge_conda.sh` | Source conda init for non-interactive shell |
| `before_conda` | `ensure_conda_env.sh` | Create conda env + install libnuma |
| `after_conda` | `install_sglang.sh` | Clone SGLang, apply CPU patches, build sgl-kernel |
| `after_conda` | `download-models.sh` | Download model weights from Hugging Face (with mirror support) |

**Modes:** `bash` = execute as script, `source` = source in current shell (exports persist).

**Stages:** `before_conda` = before `conda activate`, `after_conda` = inside activated env.

All scripts are idempotent — safe to re-run.

### Phase 3: Sweep Execution

The dispatcher constructs a single `bash -lc "..."` command per host that:

1. Acquires distributed lock (`/tmp/mmpe_scale_test_lock_fix_token_len`)
2. Kills stale processes from prior runs
3. Activates conda environment
4. Runs pre-requirements (if not skipped by resume)
5. Executes `run_scale_fix_token_len.py --no-ssh-dispatch` locally on the remote
6. Emits progress markers: `[dispatch] phase=run`, `[dispatch] sweep_pid=12345`, `[dispatch] heartbeat=N`
7. Emits final status: `[dispatch] sweep_rc=0`

### Phase 4: Result Collection

```bash
rsync -az -e "ssh [opts]" user@host:{remote_run_dir}/ {local_host_dir}/
```

- Results land in `{result_root}/{scale_id}/hosts/{host_ip}/`
- CSV path columns (`log_path`, `metrics_path`, `emon_output_path`) are rewritten from remote to local paths
- Per-host `aggregate.csv` are merged into a single top-level `aggregate.csv`
- Rsync runs even on failure (best-effort result collection)

## Resume Support

Pass `--resume` to skip completed work:

```bash
python run_scale_fix_token_len.py --config job.json --config remote.json --resume
```

Resume logic operates at three levels:

| Level | Condition | Behavior |
|-------|-----------|----------|
| **Host** | `hosts/{ip}/aggregate.csv` exists AND `sweep_rc=0` | Skip entire host |
| **Setup** | Pip marker file exists (`~/.cache/.../pip_markers/{hash}.ok`) | Skip pip install |
| **Run** | Remote run dir is non-empty | Skip pre-requirements, jump to sweep |

The pip marker key is a hash of: python path, conda env name, pip args, and requirements file contents — so any dependency change invalidates the cache.

## Monitoring

### Real-Time Monitor

```bash
bash scripts/scale-test/embedding/monitor_scale_fix_token_len.sh \
  --scale-id 20260411T120000Z \
  --interval 30
```

Displays every 30 seconds:
- Local orchestrator process liveness
- Per-host phase (`setup` / `run` / `done`)
- Per-host error count (Traceback/ERROR lines)
- Last 2 log lines per host
- Analysis artifacts generated

### Agent Skill

```bash
POST /v1/skills/scale_monitor
{"scale_id": "20260411T120000Z", "task": "embedding"}
```

Captures a single monitoring snapshot (uses 10s timeout on the infinite-loop monitor script).

## Launching

### CLI

```bash
# Embedding sweep across remote hosts
python scripts/scale-test/embedding/run_scale_fix_token_len.py \
  --config scripts/scale-test/embedding/config/local/smoke.json \
  --config scripts/scale-test/embedding/conf-remote-example.json \
  --scale-id my-run-001 \
  --tee

# Qwen3 bench_serving sweep (uses same dispatcher)
python scripts/scale-test/qwen3/run_scale_bench_serving.py \
  --config scripts/scale-test/qwen3/config/local/smoke.json \
  --config remote.json

# VL-embedding sweep
python scripts/scale-test/vl-embedding/run_scale_fix_image_size.py \
  --config scripts/scale-test/vl-embedding/config/local/smoke.json \
  --config remote.json

# With nohup (background)
bash scripts/scale-test/embedding/run_scale_fix_token_len.sh \
  --job-config job.json --remote-config remote.json --nohup
```

### Agent Skills

| Skill | Purpose |
|-------|---------|
| `remote_preflight_fix_token_len` | Verify SSH connectivity and basic host readiness |
| `scale_run_fix_token_len` | Launch or resume a sweep (pass config via `config_path`) |
| `scale_status_fix_token_len` | Query run progress: host breakdown, exit codes, aggregate counts |
| `scale_monitor` | One-shot status snapshot of a running sweep |
| `scale_analyze` | Post-hoc analysis after sweep completes |
| `log_analyze` | Pattern-match failure logs (SSH timeout, conda missing, pip error) |

## Error Handling

- **SSH failures:** Dispatcher distinguishes SSH exit code from sweep exit code via `[dispatch] sweep_rc=N` marker in logs
- **Continue-on-error:** When `continue_on_error=true`, host failures don't halt the dispatcher — all hosts proceed, failures marked in aggregate
- **SGLang clone recovery:** If `install_sglang.sh` fails due to flaky GitHub, dispatcher builds a local tarball and uploads it as fallback
- **Final exit code:** `max(all host return codes)` — any failure surfaces

## Result Directory Structure

```
result/fix_token_len/{scale_id}/
├── aggregate.csv                 # Merged results from all hosts
├── auto_test_config.generated.json
├── hosts/
│   ├── 10.0.0.11/
│   │   ├── aggregate.csv         # Per-host results
│   │   ├── remote_run.log        # Full dispatch log
│   │   └── {variant_dirs}/       # Per-variant outputs
│   └── 10.0.0.12/
│       └── ...
└── analysis/                     # Generated by scale_analyze
    ├── summary_pivot.csv
    ├── token_len_scaling.csv
    └── plot_*.png
```

## Related

- [Multi-Instance Guide](multi-instance.md) — local multi-instance (single machine)
- [Running Benchmarks](running-benchmarks.md) — general benchmark workflow
- [CPU Optimization](cpu-optimization.md) — LD_PRELOAD, AMX, NUMA settings
- [SGLang Backend](../entities/backends/sglang.md) — server configuration details
- [KV Cache](../concepts/kv-cache.md) — memory caps for multi-instance
