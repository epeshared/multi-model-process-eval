---
title: TTFT — Time To First Token
created: 2026-04-10
updated: 2026-04-10
tags: [concept, metric, latency]
---

# TTFT — Time To First Token

The elapsed time from sending a request to receiving the first output token. A key latency metric for interactive LLM applications.

## Measurement

In this project, TTFT is measured in the [Qwen3 LLM task](../entities/tasks/qwen3-llm.md):

- **vLLM HTTP streaming**: time from request send to first SSE chunk
- **SGLang HTTP**: time from request send to first response byte

## Why It Matters

- Directly impacts perceived responsiveness in chat applications
- Dominated by prompt processing (prefill) time
- Scales with input length — longer prompts → higher TTFT

## Factors Affecting TTFT

- Model size (more parameters → slower prefill)
- Input token count
- [Batch size](batch-size-tuning.md) — concurrent requests increase TTFT
- [KV cache](kv-cache.md) availability (cache hits skip prefill)
- Hardware: [AMX](amx.md) acceleration reduces compute time

## Related

- [TPOT](tpot.md) — the complementary decode-phase metric
- [Qwen3 LLM Task](../entities/tasks/qwen3-llm.md)
