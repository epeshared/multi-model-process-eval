---
title: Running Benchmarks
created: 2026-04-10
updated: 2026-04-10
tags: [guide, benchmark, workflow]
---

# Running Benchmarks

End-to-end workflow for running inference benchmarks.

## Quick Start

### Embedding (Local Torch)

No server needed:

```bash
cd scripts/embedding
MODE=input_len SYNTHETIC_INPUT_LEN=512 MAX_SAMPLES=10000 \
  BACKEND=torch DEVICE=cpu DTYPE=bfloat16 \
  ./run_fix_token_len.sh
```

### Embedding (SGLang HTTP)

1. Start server:

```bash
cd scripts/embedding/sglang
MODEL_DIR=/path/to/model BATCH_SIZE=16 PORT=30000 ./start_sglang_server.sh
```

2. Run benchmark:

```bash
cd scripts/embedding
BACKEND=sglang BASE_URL=http://127.0.0.1:30000 \
  MODE=token_len SYNTHETIC_TOKEN_LEN=64 MAX_SAMPLES=10000 \
  ./run_fix_token_len.sh
```

### LLM (Qwen3)

```bash
cd scripts/qwen3
./run_qwen3_test.sh
```

### VL (Vision-Language)

```bash
cd scripts/vl
./run_qwen_vl_flickr8k.sh
```

## Benchmark Parameters

| Variable | Purpose | Typical Values |
|----------|---------|---------------|
| `MAX_SAMPLES` | Total input samples | 1000–100000 |
| `BATCH_SIZE` | Samples per batch | 16–256 |
| `WARMUP_SAMPLES` | Warmup runs (excluded) | 1–100 |
| `MODE` | Synthetic data mode | `input_len`, `token_len` |

## Profiling

Enable torch profiler for detailed analysis:

```bash
PROFILE=1 PROFILE_ACTIVITIES=CPU PROFILE_OUT_DIR=./traces \
  ./run_fix_token_len.sh
```

Produces Chrome trace files viewable at `chrome://tracing`.

## Automated Sweeps

Use the auto-test framework for systematic parameter sweeps:

```bash
cd scripts/auto-test/embedding
python run_auto_test.py --config config_fix_token_len.json
```

Config files define: model, backend, batch sizes, token lengths, instance counts, repeat count.

Output: CSV with mean ± std across repeats.

## Tips

- Always run warmup to exclude compilation and cold-cache effects
- For [multi-instance](multi-instance.md) runs, set explicit [KV cache](../concepts/kv-cache.md) caps
- Match server `BATCH_SIZE` to client `BATCH_SIZE` for [torch compile](../concepts/torch-compile.md) efficiency

## Related

- [CPU Optimization Guide](cpu-optimization.md)
- [Multi-Instance Guide](multi-instance.md)
- [Batch Size Tuning](../concepts/batch-size-tuning.md)
