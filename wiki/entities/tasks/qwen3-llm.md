---
title: Qwen3 LLM Task
created: 2026-04-10
updated: 2026-04-10
tags: [task, llm, qwen3, text-generation]
sources: []
---

# Qwen3 LLM Task

Text generation / chat completion benchmarks using Qwen3 models.

## Entry Points

| File | Purpose |
|------|---------|
| `src/tasks/qwen3.py` | Task logic + backend dispatch |
| `scripts/qwen3/run_qwen3.py` | CLI entry point |
| `scripts/qwen3/run_qwen3_test.sh` | Quick test wrapper |
| `scripts/qwen3/run_bench_serving.sh` | Serving benchmark |

## Supported Backends

- [SGLang](../backends/sglang.md) HTTP
- [vLLM](../backends/vllm.md) HTTP (with streaming)

Backend implementations in `src/tasks/qwen3_backends/`.

## Key Metrics

| Metric | Description |
|--------|-------------|
| [TTFT](../../concepts/ttft.md) | Time to first token (latency) |
| [TPOT](../../concepts/tpot.md) | Time per output token (generation speed) |
| Total time | End-to-end request duration |
| Token usage | Prompt + completion tokens |

## Features

- Single prompt and synthetic dataset modes
- Streaming support (vLLM HTTP) for per-token timing
- Batch processing with timing aggregation

## Related

- [Qwen3 LLM Model](../models/qwen3-llm.md)
- [TTFT](../../concepts/ttft.md) | [TPOT](../../concepts/tpot.md)
