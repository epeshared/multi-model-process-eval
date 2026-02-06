# Scale Test: Embedding (fix_token_len)

This folder provides a thin “scale-test” runner that **reuses** the existing auto-test harness in
`scripts/auto-test/embedding/` to evaluate how different **token lengths** affect the performance
of `run_fix_token_len.sh`, while optionally applying **CPU core** and **memory** limits.

It also supports **Intel EMON** collection (via the auto-test runner’s `emon -collect-edp`) and
post-processing into `summary.xlsx` using:

`emon -process-pyedp /opt/intel/sep/config/edp/pyedp_config.txt`

Note: `emon -process-pyedp` runs a Python metric post-processor (`pyedp`) under whatever `python3`
is on your `PATH`. Make sure your active Python environment has the required deps installed:

- `python3 -m pip install -r requirements-emon.txt`

## Quick start

1) Edit config:

- `scripts/scale-test/embedding/config_scale_fix_token_len.json`

2) Run:

- `python3 scripts/scale-test/embedding/run_scale_fix_token_len.py --config scripts/scale-test/embedding/config_scale_fix_token_len.json --tee`

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
- `emon_metrics.csv`: extracted EMON metrics (plus `tps_per_watt`)
- `failed_variants.csv`: failures / missing summaries per variant
- `plot_tps_vs_token_len.png`
- `plot_tps_per_watt_vs_token_len.png` (only if EMON power is available)

The top-level `aggregate.csv` includes a `variant` column to distinguish combinations.

## Sweeping CPU sets / KV cache limit

You can sweep additional dimensions directly under `run`:

- `run.cpu.cpus`: either a single string (e.g. `"0-7"`) or a list (e.g. `["0-7", "8-15"]`)
- `run.sglang_max_total_tokens`: either a single value or a list (e.g. `["120000", "250000"]`)

When provided as lists, the runner executes the cartesian product:

`cpu.cpus × sglang_max_total_tokens × batch_sizes × token_lens × repeats`

## Notes on CPU/memory limiting

The runner tries to constrain the entire auto-test process tree (runner + servers + benchmark):

1) Prefer `systemd-run --user --scope` with `MemoryMax=` and `AllowedCPUs=`.
2) Fallback to `taskset -c` (CPU affinity) + `prlimit --as=` (virtual memory cap).

If neither method is available, it runs without enforcement and prints a warning.

## Continue after OOM/failures

By default, the scale-test runner stops immediately when the underlying auto-test runner returns a non-zero exit code (e.g. OOM kill / exit code 137).

If you want to **continue running later variants** (e.g. other CPU sets / other `SGLANG_MAX_TOTAL_TOKENS` values) even after a failure, set:

- `run.continue_on_error: true`

When enabled, the runner will keep going and will also write a marker row into `aggregate.csv` for failed variants (so you can see which combination failed).

## Notes on SGLang caching (important for apples-to-apples TPS)

SGLang maintains a radix/prefix cache. When you run **warmup_runs > 0** and then benchmark the
**same synthetic inputs again** (same `SYNTHETIC_SEED` / same dataset), the benchmark can become
dominated by cache hits (server logs show large `#cached-token` and tiny `#new-token`).

This effect is much stronger when **memory is not constrained**, because SGLang may allocate a very
large KV cache / memory pool based on host RAM, allowing the radix cache to retain far more prompts.

For scale-test token length sweeps we generally want to measure the **true embedding compute** under
CPU/memory limits (not warmed cache hits). The default scale-test config sets:

- `SGLANG_DISABLE_RADIX_CACHE=1`

If you intentionally want “warm cache” numbers, remove that env var.
