---
title: Qwen2.5-VL
created: 2026-04-10
updated: 2026-04-10
tags: [model, vision-language, qwen]
sources: []
---

# Qwen2.5-VL

Vision-language model from the Qwen2.5 family, supporting image + text chat.

## Variants

| Variant | Parameters |
|---------|-----------|
| Qwen2.5-VL-3B | 3B |
| Qwen2.5-VL-7B | 7B |

## Supported Backends

All five backends via the [VL task](../tasks/vl.md).

## Features

- Image + text multimodal chat
- Chat template support via `AutoModelForVision2Seq` + processor
- Profiling support (CPU/CUDA trace export)

## Usage

```bash
cd scripts/vl
./run_qwen_vl_flickr8k.sh        # Flickr8k dataset
./run_qwen_vl_synthetic.sh       # Synthetic images
```

## Related

- [VL Task](../tasks/vl.md)
- [Qwen2.5-Omni](qwen25-omni.md) — extends VL with audio
