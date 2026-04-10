---
title: Qwen3 LLM
created: 2026-04-10
updated: 2026-04-10
tags: [model, llm, qwen3, text-generation]
sources: []
---

# Qwen3 LLM

Text generation model from the Qwen3 family by Alibaba, used for chat/completion benchmarks.

## Variants

| Variant | Parameters |
|---------|-----------|
| Qwen3-0.6B | 0.6B |
| Qwen3-1.7B | 1.7B |
| Qwen3-4B | 4B |

## Supported Backends

| Backend | Status |
|---------|--------|
| [SGLang](../backends/sglang.md) HTTP | ✅ |
| [vLLM](../backends/vllm.md) HTTP | ✅ |
| torch | — |
| SGLang offline | — |
| vLLM offline | — |

## Key Metrics

- [TTFT](../../concepts/ttft.md) — Time To First Token
- [TPOT](../../concepts/tpot.md) — Time Per Output Token
- Throughput (tokens/sec)

## Usage

```bash
cd scripts/qwen3
./run_qwen3_test.sh
```

## Related

- [Qwen3 LLM Task](../tasks/qwen3-llm.md)
- [SGLang Backend](../backends/sglang.md)
