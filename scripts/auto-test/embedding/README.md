# Auto Test: Embedding & MTEB

This folder contains a lightweight runner to batch-execute embedding benchmarks across models/backends and write unified metrics (TPS/latency) into `scripts/auto-test/embedding/result/`.

## What’s inside

- `run_auto_test.py`: reads a JSON config, runs jobs, manages HTTP servers (vLLM/SGLang), parses metrics, writes logs/metrics/summary.
- `run_auto_test.sh`: convenience wrapper (lets you pass a config JSON as the first arg).
- Config examples:
  - `config_yahoo.json`
  - `config_fix_token_len.json`
  - `config_mteb.json`

## Quick start

From repo root:

- Run a config:
  - `cd scripts/auto-test/embedding`
  - `./run_auto_test.sh config_fix_token_len.json --tee`

- Run only selected jobs (exact name match; repeatable):
  - `./run_auto_test.sh config_mteb.json --tee --only mteb_STSBenchmark_qwen3-embedding-4b_sglang-http`

- Skip specific jobs (exact name match; repeatable):
  - `./run_auto_test.sh config_mteb.json --tee --skip jobA --skip jobB`

## Output layout

For each run, a `run_id` directory is created:

- Per-run artifacts:
  - `scripts/auto-test/embedding/result/<suite>/<run_id>/`
    - `*_job.log` / `*.warmup_XX.log`: raw stdout (and stderr merged)
    - `*.metrics.json`: parsed metrics + metadata
    - `summary_<run_id>.jsonl`: one JSON record per job
    - `*_server_*.log`: server logs (when the runner started the server)

- Summary CSV (always in the suite root):
  - `scripts/auto-test/embedding/result/<suite>/summary_<run_id>.csv`

### CSV columns

The CSV contains one row per job. It includes:

- job identity: `job_name`, `script`, `backend`, `model`, `model_id`
- server NUMA binding (effective): `numactl_cores`, `numactl_cpunodebind`, `numactl_membind`
- performance: `tps`, `latency_sec`, `avg_batch_time_sec`, `count`, `num_batches`
- bookkeeping: `exit_code`, timestamps, log/metrics paths

## Config format (high level)

A config JSON contains:

- `script_aliases`: maps a script key (e.g. `run_fix_token_len`) to the actual `.sh` path.
- `defaults`: default `env`, timeouts, result_dir, and optional behaviors.
- `servers`: how to start servers for `sglang` and/or `vllm-http` backends.
- `jobs`: list of jobs; each job defines `name`, `script`, `args`, `env`, etc.

### Job args

`jobs[].args` supports:

- legacy positional list: `"args": ["20"]`
- named object (recommended):
  - fix token length: `"args": {"token_len": 20}`
  - MTEB: `"args": {"task": "STSBenchmark"}`

## Server lifecycle

If a job uses an HTTP backend (e.g. `BACKEND=sglang` or `BACKEND=vllm-http`) and sets `BASE_URL`, the runner can manage the server:

- starts it using `servers[backend].start_script`
- waits until it’s ready using the configured readiness probes
- tears down servers at the end of the whole run

### Restart servers (optional)

If you want the runner to kill any existing listener on the configured port before starting a new server:

- CLI: `--restart-servers`
- Config:
  - `defaults.restart_servers: true`
  - or per job: `jobs[].restart_servers: true`

### Stop server after each job (recommended when MODEL_DIR differs per job)

When multiple jobs share the same port but need different `MODEL_DIR` / served model name, enable per-job shutdown:

- CLI: `--stop-servers-after-job`
- Config:
  - `defaults.stop_server_after_job: true`
  - or per job: `jobs[].stop_server_after_job: true`

This only stops servers started by the runner (it won’t kill an “external” server).

## Warmup

Each job may define `warmup_runs` (integer). Warmups:

- run the same command N times
- write `*.warmup_XX.log`
- do NOT contribute to summary metrics
- if any warmup fails, the runner stops the run

## Re-parse an existing run

If you still have the per-run `*.metrics.json` and logs, you can regenerate summary files:

- `python3 scripts/auto-test/embedding/run_auto_test.py --config scripts/auto-test/embedding/config_fix_token_len.json --reparse-run-id <RUN_ID>`

This rewrites:

- `result/<suite>/<run_id>/summary_<run_id>.jsonl`
- `result/<suite>/summary_<run_id>.csv`

## Troubleshooting

- **MTEB: `ModuleNotFoundError: No module named 'src'`**
  - `scripts/embedding/mteb/run_mteb.sh` sets `PYTHONPATH` to repo root to avoid issues with `PYTHONSAFEPATH`/isolated environments.

- **Offline backends unexpectedly hit HuggingFace**
  - For offline runs (e.g. `sglang-offline`), ensure `MODEL_ID` is a local path (or set `MODEL_DIR`).

- **No numactl info in CSV**
  - The CSV records the *effective server binding* (job override > server defaults). Ensure `servers.<backend>.numactl.*` is set if you don’t set `NUMACTL_*` in the job env.
