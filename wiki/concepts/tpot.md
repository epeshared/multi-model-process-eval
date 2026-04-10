---
title: TPOT — Time Per Output Token
created: 2026-04-10
updated: 2026-04-10
tags: [concept, metric, throughput]
---

# TPOT — Time Per Output Token

The average time to generate each output token after the first. Measures decode-phase throughput.

## Measurement

In this project, TPOT is measured in the [Qwen3 LLM task](../entities/tasks/qwen3-llm.md):

- Computed as `(total_time - ttft) / (output_tokens - 1)`
- Only meaningful for streaming responses with >1 output token

## Why It Matters

- Determines text generation speed experienced by the user
- Lower TPOT = faster token streaming
- Relatively stable across different input lengths (unlike [TTFT](ttft.md))

## Factors Affecting TPOT

- Model size
- [KV cache](kv-cache.md) memory bandwidth
- [Batch size](batch-size-tuning.md) — batching can amortize overhead
- Hardware memory bandwidth (often the bottleneck on CPU)

## Related

- [TTFT](ttft.md) — the complementary prefill-phase metric
- [Qwen3 LLM Task](../entities/tasks/qwen3-llm.md)
