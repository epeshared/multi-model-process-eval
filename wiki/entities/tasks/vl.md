---
title: VL Task
created: 2026-04-10
updated: 2026-04-10
tags: [task, vision-language, multimodal]
sources: []
---

# VL (Vision-Language) Task

Image + text multimodal chat benchmarks using Qwen2.5-VL models.

## Entry Points

| File | Purpose |
|------|---------|
| `src/tasks/vl.py` | Task logic + backend dispatch |
| `scripts/vl/run_vl.py` | CLI entry point |
| `scripts/vl/run_qwen_vl_flickr8k.sh` | Flickr8k benchmark |
| `scripts/vl/run_qwen_vl_synthetic.sh` | Synthetic image benchmark |

## Supported Backends

All five: [torch](../backends/torch.md), [SGLang](../backends/sglang.md) (HTTP + offline), [vLLM](../backends/vllm.md) (HTTP + offline).

Backend implementations in `src/tasks/vl_backends/`.

## Dataset Support

- **Single image**: one image + prompt
- **Flickr8k**: image + caption pairs for batch testing
- **Synthetic**: generated images with configurable H×W

## Profiling

```bash
PROFILE=1 PROFILE_ACTIVITIES=CPU PROFILE_OUT_DIR=./traces \
  ./scripts/vl/run_qwen_vl_synthetic.sh
```

Exports Chrome traces for analysis.

## Related

- [Qwen2.5-VL Model](../models/qwen25-vl.md)
- [Omni Task](omni.md) — extends VL with audio
- [Embedding Task](embedding.md) — related multimodal task
