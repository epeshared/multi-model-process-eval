# Scale Test: Embedding (fix_token_len)

This folder provides a thin “scale-test” runner that **reuses** the existing auto-test harness in
`scripts/auto-test/embedding/` to evaluate how different **token lengths** affect the performance
of `run_fix_token_len.sh`, while optionally applying **CPU core** and **memory** limits.

It also supports **Intel EMON** collection (via the auto-test runner’s `emon -collect-edp`) and
post-processing into `summary.xlsx` using:

`emon -process-pyedp /opt/intel/sep/config/edp/pyedp_config.txt`

## Quick start

1) Edit config:

- `scripts/scale-test/embedding/config_scale_fix_token_len.json`

2) Run:

- `python3 scripts/scale-test/embedding/run_scale_fix_token_len.py --config scripts/scale-test/embedding/config_scale_fix_token_len.json --tee`

## Outputs

Each run creates a folder like:

- `scripts/scale-test/embedding/result/fix_token_len/<scale_id>/`

Inside it you’ll find:

- `auto_test_config.generated.json`: the generated auto-test config fed into `run_auto_test.py`
- `auto_test_stdout.log`: stdout/stderr from the auto-test runner
- `summary_<run_id>.csv`: per-job TPS/latency/etc (written by `run_auto_test.py`)
- `<run_id>/...`: per-job logs + `*.metrics.json` (+ `*.emon/emon.dat` when enabled)
- `aggregate.csv`: merged perf + emon paths (and emon key-values if parsable)

## Notes on CPU/memory limiting

The runner tries to constrain the entire auto-test process tree (runner + servers + benchmark):

1) Prefer `systemd-run --user --scope` with `MemoryMax=` and `AllowedCPUs=`.
2) Fallback to `taskset -c` (CPU affinity) + `prlimit --as=` (virtual memory cap).

If neither method is available, it runs without enforcement and prints a warning.
