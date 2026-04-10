---
title: Qwen3-Embedding
created: 2026-04-10
updated: 2026-04-10
tags: [model, embedding, qwen3]
sources: []
---

# Qwen3-Embedding

Text embedding model from the Qwen3 family by Alibaba.

## Variants

| Variant | Parameters | Default Path |
|---------|-----------|--------------|
| Qwen3-Embedding-0.6B | 0.6B | `Qwen/Qwen3-Embedding-0.6B` |
| Qwen3-Embedding-4B | 4B | `Qwen/Qwen3-Embedding-4B` |

## Supported Backends

All five backends: [torch](../backends/torch.md), [SGLang](../backends/sglang.md) (HTTP + offline), [vLLM](../backends/vllm.md) (HTTP + offline).

## Usage

Typically run via the [Embedding task](../tasks/embedding.md):

```bash
MODEL=qwen3-embedding-4b MODEL_ID=/path/to/Qwen3-Embedding-4B \
  BACKEND=sglang BASE_URL=http://127.0.0.1:30000 \
  ./scripts/embedding/run_fix_token_len.sh
```

## Configuration Notes

- Supports `--trust-remote-code` flag
- Default dtype: `bfloat16`
- With [torch compile](../../concepts/torch-compile.md): `--enable-torch-compile --torch-compile-max-bs <N>`
- Attention backend: `--attention-backend intel_amx` on CPU (see [AMX](../../concepts/amx.md))

## Related

- [Embedding Task](../tasks/embedding.md)
- [SGLang Backend](../backends/sglang.md)
- [Torch Compile](../../concepts/torch-compile.md)
