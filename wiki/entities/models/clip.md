---
title: CLIP
created: 2026-04-10
updated: 2026-04-10
tags: [model, embedding, multimodal, openai]
sources: []
---

# CLIP

OpenAI's Contrastive Language–Image Pretraining model, used for multimodal (text + image) embedding.

## Variants

| Variant | Resolution | Default Path |
|---------|-----------|--------------|
| clip-vit-base-patch32 | 224×224 | `openai/clip-vit-base-patch32` |
| clip-vit-large-patch14-336 | 336×336 | `openai/clip-vit-large-patch14-336` |

## Supported Backends

All five backends via the [Embedding task](../tasks/embedding.md).

## Usage

Primarily used with the Flickr8k benchmark:

```bash
MODEL=clip-vit-base-patch32 BACKEND=torch \
  ./scripts/embedding/run_embedding_flickr8k.sh
```

## Notes

- Supports both text and image embedding in a shared vector space
- Image input via data-url, base64, or file path depending on backend
- The [torch backend](../backends/torch.md) uses `transformers` CLIPModel directly

## Related

- [Embedding Task](../tasks/embedding.md)
- [Youtu-Embedding](youtu-embedding.md) — alternative embedding model
