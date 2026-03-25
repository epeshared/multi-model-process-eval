# Scale Test: Qwen3 Bench Serving

This folder adds Qwen3 text-generation scale tests to the existing scale-test workflow by wrapping:

- `scripts/qwen3/run_bench_serving.sh`
- `python -m sglang.bench_serving`

The default local target is `Qwen/Qwen3-0.6B`.

## Quick start

1. Start a Qwen3 SGLang server via scale-test, using the smoke config:

   ```bash
   bash scripts/scale-test/qwen3/run_scale_bench_serving.sh \
     --job-config scripts/scale-test/qwen3/config/local/smoke.json --tee
   ```

2. Run a broader sweep over input length and max concurrency:

   ```bash
   bash scripts/scale-test/qwen3/run_scale_bench_serving.sh \
     --job-config scripts/scale-test/qwen3/config/local/local-sglang.json --tee
   ```

## What is swept

- `run.sweep_env_key = RANDOM_INPUT_LEN`
- `run.bench.batch_env_key = MAX_CONCURRENCY`

This means the generic runner reuses:

- `sweep_values` for `--random-input-len`
- `batch_sizes` for `--max-concurrency`

while keeping server-side `BATCH_SIZE` independent in `server_template.batch_size`.

## Benchmark shape

The wrapped benchmark is equivalent to:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 3000 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --random-range-ratio 0.5
```

with optional `--max-concurrency` driven by scale-test.

## Outputs

Results are written under:

- `scripts/scale-test/qwen3/result/bench_serving*/<scale_id>/`

Each job emits:

- auto-test summary CSV/JSONL
- per-job logs
- aggregate CSV

Key metrics carried through the pipeline include:

- `output_throughput` mapped to `tps`
- `request_throughput` mapped to `qps`
- `mean_ttft_ms`
- `mean_tpot_ms`
- `mean_e2e_latency_ms`