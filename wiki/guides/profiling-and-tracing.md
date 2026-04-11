---
title: Profiling & Tracing
created: 2026-04-11
updated: 2026-04-11
tags: [guide, profiling, torch, sglang, tracing]
sources: [src/tasks/embedding.py, src/tasks/vl.py, scripts/embedding/sglang/start_sglang_server.sh]
---

# Profiling & Tracing

Two profiling systems are available: PyTorch Profiler (client-side) and SGLang Server Profiler (server-side).

## PyTorch Profiler

### Enable

```bash
PROFILE=1 PROFILE_ACTIVITIES=CPU PROFILE_OUT_DIR=./traces \
  ./run_fix_token_len.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROFILE` | `0` | Enable profiler (`1` to enable) |
| `PROFILE_ACTIVITIES` | `CPU` | Profiler activities: `CPU`, `CUDA`, or both |
| `PROFILE_RECORD_SHAPES` | `0` | Record tensor shapes (adds overhead) |
| `PROFILE_OUT_DIR` | `./profile_traces` | Output directory for trace files |

### Output

Produces Chrome Trace JSON files (`*.json` or `*.json.gz`):

```
profile_traces/
├── embedding_torch_batch0.json
├── embedding_torch_batch1.json
└── ...
```

**Viewing:** Open `chrome://tracing` (or `edge://tracing`) in browser and load the trace file. Shows:
- Per-operator timing (ms)
- Operator call stacks
- CPU thread activity
- Tensor shapes (if `PROFILE_RECORD_SHAPES=1`)

### Which Tasks Support It

| Task | Torch | SGLang HTTP | vLLM |
|------|:-----:|:-----------:|:----:|
| Embedding | ✅ | — | — |
| VL | ✅ | ✅ (via server profiler) | — |
| Qwen3 LLM | — | — | — |
| Omni | — | — | — |

Client-side torch profiling is only meaningful for the `torch` backend (local inference). For HTTP backends, use the server-side profiler.

## SGLang Server Profiler

### Enable

SGLang exposes HTTP-triggered profiling via `/start_profile` and `/stop_profile` endpoints.

```bash
# Start profiling
curl -X POST http://127.0.0.1:30000/start_profile

# Run your benchmark...

# Stop profiling
curl -X POST http://127.0.0.1:30000/stop_profile
```

### Server-Side Config

Set the profile output directory before starting the server:

```bash
export SGLANG_TORCH_PROFILER_DIR="$PWD/sglang_logs/sglang_cpu"
```

Traces are written to this directory after `/stop_profile` is called.

### Programmatic Access

The VL task supports automated profiling via the `profile` parameter:

```python
# In vl_backends/sglang_http.py
session.start_profile()   # POST /start_profile
# ... run inference ...
session.stop_profile()    # POST /stop_profile
```

Via agent skill:
```json
POST /v1/skills/vl_chat
{"args": {"model": "Qwen2.5-VL-3B", "backend": "sglang", "profile": true}}
```

### Custom Profile Paths

Override default paths via kwargs:

```python
session.start_profile(start_path="/custom_start_profile")
session.stop_profile(stop_path="/custom_stop_profile")
```

## EMON (Hardware Counters)

For CPU hardware-level profiling (power, cache, TMA), see [EMON](../concepts/emon.md).

## Tips

- **Warmup first:** Always run warmup samples before profiling to exclude [torch compile](../concepts/torch-compile.md) compilation time
- **Match batch sizes:** Profile with the same batch size used in production for relevant traces
- **Minimize overhead:** Avoid `PROFILE_RECORD_SHAPES=1` for throughput benchmarks — adds 5-15% overhead
- **Server profiling:** SGLang `/start_profile` captures server-side kernels including [AMX](../concepts/amx.md) operations that client profiling can't see

## Related

- [EMON](../concepts/emon.md) — hardware performance counters
- [CPU Optimization Guide](cpu-optimization.md) — AMX, IPEX flags
- [Torch Compile](../concepts/torch-compile.md) — compilation impact on profiling
- [Auto-Test Framework](auto-test-framework.md) — automated EMON collection
