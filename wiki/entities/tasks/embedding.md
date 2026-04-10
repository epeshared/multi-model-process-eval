---
title: Embedding Task
created: 2026-04-10
updated: 2026-04-10
tags: [task, embedding, text, image]
sources: []
---

# Embedding Task

Text and image embedding — the most feature-complete task in the framework.

## Entry Points

| File | Purpose |
|------|---------|
| `src/tasks/embedding.py` | Task logic + backend dispatch |
| `scripts/embedding/run_embedding.py` | CLI entry point |
| `scripts/embedding/run_fix_token_len.sh` | Synthetic fixed-length benchmark |
| `scripts/embedding/run_embedding_yahoo.sh` | Yahoo Answers benchmark |
| `scripts/embedding/run_embedding_flickr8k.sh` | Flickr8k benchmark |

## Supported Backends

All five: [torch](../backends/torch.md), [SGLang](../backends/sglang.md) HTTP, SGLang offline, [vLLM](../backends/vllm.md) HTTP, vLLM offline.

Backend implementations in `src/tasks/embedding_backends/`.

## Data Sources

- **Synthetic**: fixed token-length or character-length generated texts
- **Yahoo Answers**: real-world text from JSONL (questions, answers, or both)
- **Flickr8k**: image + caption pairs
- **Custom**: any text file, JSONL, or image directory

## Synthetic Modes

| MODE | Parameter | Description |
|------|-----------|-------------|
| `token_len` | `SYNTHETIC_TOKEN_LEN` | N samples of M word-tokens |
| `input_len` | `SYNTHETIC_INPUT_LEN` | N samples of K characters |

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BACKEND` | varies | Backend selection |
| `MODEL` / `MODEL_ID` | — | Model name / path |
| `BATCH_SIZE` | 100 | Embedding batch size |
| `MAX_SAMPLES` | 1000 | Max input samples |
| `WARMUP_SAMPLES` | 1 | Warmup runs (excluded from timing) |
| `DEVICE` | cpu | Device |
| `DTYPE` | bfloat16 | Precision |

## Metrics

- Throughput: samples/sec
- Average batch time (ms)
- Total elapsed time

## Related

- [Models](../models/qwen3-embedding.md) | [CLIP](../models/clip.md) | [Youtu](../models/youtu-embedding.md)
- [Batch Size Tuning](../../concepts/batch-size-tuning.md)
- [Running Benchmarks](../../guides/running-benchmarks.md)
