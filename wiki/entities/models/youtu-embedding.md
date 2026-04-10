---
title: Youtu-Embedding
created: 2026-04-10
updated: 2026-04-10
tags: [model, embedding, tencent]
sources: []
---

# Youtu-Embedding

Tencent Youtu text embedding model (FP16 variant).

## Details

| Property | Value |
|----------|-------|
| Provider | Tencent |
| Default Path | `tencent/youtu-embedding-fp16` |
| Precision | FP16 (also usable as BF16) |

## Supported Backends

All five backends via the [Embedding task](../tasks/embedding.md).

## Usage

Default model in the SGLang embedding server script:

```bash
# Uses youtu-embedding-fp16 by default
cd scripts/embedding/sglang
./start_sglang_server.sh
```

Override with `MODEL_DIR`:

```bash
MODEL_DIR=/path/to/youtu-embedding-fp16 ./start_sglang_server.sh
```

## Related

- [Embedding Task](../tasks/embedding.md)
- [Qwen3-Embedding](qwen3-embedding.md) — alternative embedding model
- [CLIP](clip.md) — multimodal alternative
