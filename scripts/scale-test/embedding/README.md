# Scale Test: Embedding (fix_token_len)

This folder provides a thin “scale-test” runner that **reuses** the existing auto-test harness in
`scripts/auto-test/embedding/` to evaluate how different **token lengths** affect the performance
of `run_fix_token_len.sh`, while optionally applying **CPU core** and **memory** limits.

It also supports **Intel EMON** collection (via the auto-test runner’s `emon -collect-edp`) and
post-processing into `summary.xlsx` using:

`emon -process-pyedp /opt/intel/sep/config/edp/pyedp_config.txt`

EMON config knobs:

- Enable EMON collection per job via `run.job_template.emon.emon_enable`.
- Enable post-processing via `run.job_template.emon.process_after_run`.

Note: `emon -process-pyedp` runs a Python metric post-processor (`pyedp`) under whatever `python3`
is on your `PATH`. Make sure your active Python environment has the required deps installed:

- `python3 -m pip install -r requirements-emon.txt`

## Quick start

1) Edit config:

- Job/run config: `scripts/scale-test/embedding/config/conf-job.json`
- Optional remote/server config: `scripts/scale-test/embedding/config/conf-remote.json`

2) Run:

- `conda activate sglang-cpu`

- `python3 scripts/scale-test/embedding/run_scale_fix_token_len.py --job-config scripts/scale-test/embedding/config/conf-job.json --tee`

### Split config (job/run vs remote/servers)

If you want to keep sweep/job settings separate from SSH/remote server settings, use:

- Job config (sweep + server_template + emon): `scripts/scale-test/embedding/config/conf-job.json`
- Remote config (servers + ssh + remote_repo_dir): `scripts/scale-test/embedding/config/conf-remote.json`

Run with two configs:

- `python3 scripts/scale-test/embedding/run_scale_fix_token_len.py --job-config scripts/scale-test/embedding/config/conf-job.json --remote-config scripts/scale-test/embedding/config/conf-remote.json --tee`

Notes:

- `--remote-config` is optional (useful for local-only runs).
- If a server `ip` is `127.0.0.1` / `localhost`, the dispatch code treats it as local and does not require SSH.

Device selection:

- Set `server_template.device` to `"cpu"` or `"cuda"` in your scale-test JSON.
	- `cuda` uses `scripts/embedding/sglang/start_sglang_server_cuda.sh`
	- `cpu` uses `scripts/embedding/sglang/start_sglang_server.sh`

Python selection (`SGLANG_PYTHON`):

- You can omit `SGLANG_PYTHON` in JSON. When starting an SGLang server, the auto-test runner defaults it to its own interpreter (`sys.executable`).
- Local runs still need you to execute the runner under the right env (e.g. `conda run -n <env> python ...` / `conda run -n <env> bash ...`), otherwise the default interpreter may not have `sglang` installed.
- Optional (JSON-only, portable): set `server_template.conda_env` to a conda env name (e.g. `"xtang-embedding-cuda"`). This will set `SGLANG_CONDA_ENV` and the server start script will run via `conda run -n <env> python ...`.

## Generate local test images (different resolutions)

For image-embedding benchmarks, you can generate a deterministic set of local images with arbitrary
resolutions (M×N) and avoid relying on external URLs:

- `python3 scripts/scale-test/embedding/gen_test_images.py --out /tmp/mmpe_imgs --sizes 224x224,384x384,512x512,1024x1024,1280x720,1920x1080 --per-size 4`

This produces files like `img_1280x720_00012.png` under the output folder.

Notes:

- Pillow is already included in `requirements-cpu.txt` and `requirements-cuda.txt`.
- For throughput/latency tests, synthetic images (checker/gradient/noise) are usually sufficient.

### Resume an interrupted run (skip completed)

If a run was interrupted (SSH disconnect, reboot, etc), you can re-run the same
`scale_id` and have the runner only execute the missing work:

- `python3 scripts/scale-test/embedding/run_scale_fix_token_len.py --job-config scripts/scale-test/embedding/config/conf-job.json --remote-config scripts/scale-test/embedding/config/conf-remote.json --scale-id <scale_id> --resume --tee`

Behavior:

- Local (no dispatch): reads existing `summary_*.csv` under the variant dir and only runs jobs that do **not** have a successful (`exit_code=0`) row.
- Multi-host dispatch: if the local copied-back host folder already contains a successful run marker (host `aggregate.csv` exists and `remote_run.log` ends with `sweep_rc=0`), that host is skipped.

### Background (nohup) run

If you want the runner to keep running after you disconnect, use the wrapper script’s `--nohup` mode.
It writes the local launcher logs under the matching run directory:

- `<result_root>/<scale_id>/launcher_logs/nohup.log`
- `<result_root>/<scale_id>/launcher_logs/nohup.pid`

Example:

- `bash scripts/scale-test/embedding/run_scale_fix_token_len.sh --job-config scripts/scale-test/embedding/config/conf-job.json --remote-config scripts/scale-test/embedding/config/conf-remote.json --nohup`
- `bash scripts/scale-test/embedding/monitor_scale_fix_token_len.sh --scale-id <scale_id>`

## Outputs

Each run creates a folder like:

- `scripts/scale-test/embedding/result/fix_token_len/<scale_id>/`

If you sweep multiple CPU sets and/or multiple KV limits, the runner creates per-variant subfolders:

- `.../<scale_id>/cpu_<cpus>__kv_<max_total_tokens>/`
- `.../<scale_id>/variants.json` (manifest)

Inside it you’ll find:

- `auto_test_config.generated.json`: the generated auto-test config fed into `run_auto_test.py`
- `auto_test_stdout.log`: stdout/stderr from the auto-test runner
- `summary_<run_id>.csv`: per-job TPS/latency/etc (written by `run_auto_test.py`)
- `<run_id>/...`: per-job logs + `*.metrics.json` (+ `*.emon/emon.dat` when enabled)
- `aggregate.csv`: merged perf + emon paths (and emon key-values if parsable)

## Analyze a completed run

Given a run directory like:

- `scripts/scale-test/embedding/result/fix_token_len/<scale_id>/`

You can generate post-hoc analysis artifacts (CSVs + plots) via:

- `python3 scripts/scale-test/embedding/analyze_run.py scripts/scale-test/embedding/result/fix_token_len/<scale_id>/`

This writes to `<run_dir>/analysis/`:

- `summary_pivot.csv`: wide pivot table (TPS/latency per token_len × batch_size)
- `emon_metrics.csv`: extracted EMON metrics (if present)
- `emon_socket_metrics.csv`: extracted per-socket EMON metrics from `summary.xlsx` "socket view"
- `failed_variants.csv`: failures / missing summaries per variant
- `token_len_scaling.csv`: token length scalability (includes derived `tokens_per_sec`)
- `batch_size_scaling.csv`: batch size scalability
- `cpu_scaling.csv`: CPU core scalability
- `kv_scaling.csv`: KV cap (`SGLANG_MAX_TOTAL_TOKENS`) scalability
- `plot_token_len_scaling.png`
- `plot_batch_size_scaling.png`
- `plot_cpu_scaling.png`
- `plot_kv_cap_scaling.png`

The top-level `aggregate.csv` includes a `variant` column to distinguish combinations.

## Sweeping CPU sets / KV cache limit

You can sweep additional dimensions under `run.bench`:

- `run.bench.cpu.cpus`: either a single string (e.g. `"0-7"`) or a list (e.g. `["0-7", "8-15"]`)
- `run.bench.sglang_max_total_tokens`: either a single value or a list (e.g. `["120000", "250000"]`)

When provided as lists, the runner executes the cartesian product:

`bench.cpu.cpus × bench.sglang_max_total_tokens × bench.batch_sizes × bench.token_lens × repeats`

## Multi-host (SSH dispatch)

If you want to run the **same sweep on multiple servers** and copy the results back into the local
`result_root`, add a server list under `run`:

```json
{
	"run": {
		"servers": [
			{
				"ip": "10.0.0.11",
				"username": "ubuntu",
				"password": "",
				"remote_repo_dir": "/path/to/multi-model-process-eval",
				"pre_requirements": [
					{
						"file": "scripts/scale-test/pre-requirements/install_miniforge3_linux_x86_64.sh",
						"stage": "before_conda",
						"mode": "bash"
					},
					{
						"file": "scripts/scale-test/pre-requirements/install_sglang.sh",
						"stage": "after_conda",
						"mode": "bash"
					}
				],
				"requirements_profile": "cpu",
				"conda_env": "xtang-embedding-cpu",
				"install_requirements": false,
				"requirements_files": ["requirements.txt"],
				"pip_extra_args": [],
				"pre_setup_cmds": []
			},
			{
				"ip": "10.0.0.12",
				"username": "ubuntu",
				"remote_repo_dir": "/path/to/multi-model-process-eval",
				"requirements_profile": "cuda",
				"remote_python": ["conda", "run", "-n", "xtang-embedding-cuda", "python"],
				"install_requirements": true,
				"requirements_files": ["requirements.txt", "requirements-mteb.txt"]
			}
		],

		"remote_result_root": "/path/to/result/fix_token_len",
		"install_requirements": false,
		"requirements_profile": "cpu",
		"requirements_files": [],
		"pip_extra_args": [],

		"ssh": {
			"user": "", 
			"port": 22,
			"identity_file": "",
			"options": [],
			"password_file": "scripts/scale-test/embedding/passwords.json"
		}
	}
}
```

Backward compatibility:

- You can still use `"servers": ["10.0.0.11", "10.0.0.12"]` with the global `run.remote_*` and `run.ssh` defaults.

`remote_python` can be a list if you need an environment wrapper, e.g.

- `"remote_python": ["conda", "run", "-n", "myenv", "python"]`

If you set `password`, the dispatcher uses `sshpass` on the **local** machine for ssh/scp/rsync.
Storing passwords in plain JSON is not recommended; prefer SSH keys.

Password file support:

- If `servers[*].password` is `null`, the runner will read that user's password from a local password file.
- Default path: `scripts/scale-test/embedding/passwords.json` (ignored by git).
- You can override globally via `run.ssh.password_file`, or per-server via `servers[*].password_file`.

Example `passwords.json`:

```json
{
	"users": {
		"ubuntu": "your-ssh-password",
		"root": "root-password"
	},
	"hosts": {
		"10.0.0.11": {
			"ubuntu": "host-specific-password"
		},
		"10.0.0.12": "fallback-for-any-user"
	}
}
```

Then run as usual:

- `python3 scripts/scale-test/embedding/run_scale_fix_token_len.py --job-config scripts/scale-test/embedding/config/conf-job.json --remote-config scripts/scale-test/embedding/config/conf-remote.json --tee`

Notes / assumptions:

- Each server must be reachable via `ssh`.
- Requires `scp` and `rsync` on the local machine.
- If you set `password`, requires `sshpass` on the local machine (non-interactive).
- If you omit `password`, ssh/scp/rsync may prompt interactively in your terminal when needed.
- `remote_repo_dir` must exist on each remote host and contain this repo (so relative paths in the config still work).
- The remote run directory is `<remote_result_root>/<scale_id>/`.

Pre-requirements (optional):

- Prefer `pre_requirements` (a list) for multi-step bootstrap.

Stages:

- `before_conda`: runs before any `conda activate`.
- `after_conda`: runs after `eval "$(conda shell.bash hook)"; conda activate <conda_env>;` in the **same remote shell**.

Modes:

- `mode: "bash"` runs `bash <file>`.
- `mode: "source"` runs `. <file>` (useful if the script must `export` env vars for later commands).

Legacy compatibility:

- `pre_requirements_file` + `pre_requirements_use_conda_env` are still accepted.

Outputs:

- Local root: `<result_root>/<scale_id>/`
- Per-host copy: `<result_root>/<scale_id>/hosts/<host>/...`
- Combined aggregate: `<result_root>/<scale_id>/aggregate.csv` with an extra `server_host` column.

## Notes on CPU/memory limiting

The runner tries to constrain the entire auto-test process tree (runner + servers + benchmark):

1) Prefer `systemd-run --user --scope` with `MemoryMax=` and `AllowedCPUs=`.
2) Fallback to `taskset -c` (CPU affinity) + `prlimit --as=` (virtual memory cap).

If neither method is available, it runs without enforcement and prints a warning.

## Continue after OOM/failures

By default, the scale-test runner stops immediately when the underlying auto-test runner returns a non-zero exit code (e.g. OOM kill / exit code 137).

If you want to **continue running later variants** (e.g. other CPU sets / other `SGLANG_MAX_TOTAL_TOKENS` values) even after a failure, set:

- `run.job_template.continue_on_error: true`

When enabled, the runner will keep going and will also write a marker row into `aggregate.csv` for failed variants (so you can see which combination failed).

## Notes on SGLang caching (important for apples-to-apples TPS)

SGLang maintains a radix/prefix cache. When you run **run.job_template.warmup_runs > 0** and then benchmark the
**same synthetic inputs again** (same `SYNTHETIC_SEED` / same dataset), the benchmark can become
dominated by cache hits (server logs show large `#cached-token` and tiny `#new-token`).

This effect is much stronger when **memory is not constrained**, because SGLang may allocate a very
large KV cache / memory pool based on host RAM, allowing the radix cache to retain far more prompts.

For scale-test token length sweeps we generally want to measure the **true embedding compute** under
CPU/memory limits (not warmed cache hits). The default scale-test config sets:

- `SGLANG_DISABLE_RADIX_CACHE=1`

If you intentionally want “warm cache” numbers, remove that env var.
