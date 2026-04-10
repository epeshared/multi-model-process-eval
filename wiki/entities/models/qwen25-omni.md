---
title: Qwen2.5-Omni
created: 2026-04-10
updated: 2026-04-10
tags: [model, multimodal, omni, qwen]
sources: []
---

# Qwen2.5-Omni

Multimodal model from the Qwen2.5 family, supporting image + text (and audio in future).

## Variants

| Variant | Parameters |
|---------|-----------|
| Qwen2.5-Omni-3B | 3B |
| Qwen2.5-Omni-7B | 7B |

## Supported Backends

| Backend | Status |
|---------|--------|
| [SGLang](../backends/sglang.md) HTTP | ✅ |
| [vLLM](../backends/vllm.md) offline | ✅ |
| [vLLM](../backends/vllm.md) HTTP | ✅ |
| torch | — |
| SGLang offline | — |

## Usage

```bash
cd scripts/omni
./run_qwen_omni_synthetic.sh
```

## Notes

- Currently image-dominant in the benchmark scripts
- Uses OpenAI-compatible multimodal chat API for HTTP backends

## Related

- [Omni Task](../tasks/omni.md)
- [Qwen2.5-VL](qwen25-vl.md) — vision-language sibling
