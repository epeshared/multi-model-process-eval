---
title: VL-Embedding Task
created: 2026-04-11
updated: 2026-04-11
tags: [task, embedding, image, vl-embedding, multimodal]
sources: [scripts/vl-embedding/run_image_embedding.py, scripts/vl-embedding/run_fix_image_size.sh]
---

# VL-Embedding Task

Image-only embedding benchmarks via HTTP server (SGLang or vLLM). Unlike the text [Embedding](embedding.md) task, this focuses on encoding images into vector representations for downstream retrieval or similarity tasks.

## Supported Backends

| Backend | Endpoint | Notes |
|---------|----------|-------|
| `sglang` | `/v1/embeddings` | Requires `--is-embedding --enable-multimodal` |
| `vllm-http` | `/v1/embeddings` | Requires `SERVED_MODEL_NAME` |

Torch and offline backends are not supported — image embedding requires a running HTTP server.

## Entry Points

| File | Purpose |
|------|---------|
| `scripts/vl-embedding/run_image_embedding.py` | Python client — sends images to server, measures throughput |
| `scripts/vl-embedding/run_fix_image_size.sh` | Shell wrapper with env var configuration |

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backend` | `sglang` | `sglang` or `vllm-http` |
| `--model-id` | (required) | Served model name on server |
| `--base-url` | (required) | HTTP server URL |
| `--images-dir` | (required) | Directory containing test images |
| `--image-size` | (none) | Filter tag in filename (e.g. `512x512`) |
| `--batch-size` | 32 | Images per batch |
| `--max-samples` | 1000 | Max images to embed |
| `--warmup-samples` | 1 | Warmup count (excluded from metrics) |
| `--image-transport` | `data-url` | `data-url`, `base64`, `path`, or `url` |
| `--normalize` | true | L2 normalize embeddings |
| `--api` | `v1` | API style: `v1` or `openai` |
| `--timeout` | 900 | HTTP timeout (seconds) |

## Environment Variables (Shell Wrapper)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `qwen3-vl-embedding-2b` | Logical model key |
| `MODEL_ID` | `Qwen3-VL-Embedding-2B` | Served model name |
| `BACKEND` | `sglang` | Backend type |
| `BASE_URL` | `http://127.0.0.1:30000` | Server URL |
| `IMAGE_DIR` | (required) | Image directory |
| `IMAGE_SIZE` | (none) | Resolution filter |
| `IMAGE_TRANSPORT` | `data-url` | Transport mode |
| `BATCH_SIZE` | 32 | Batch size |
| `MAX_SAMPLES` | 1000 | Max images |
| `WARMUP_SAMPLES` | 1 | Warmup count |

## Output Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `tps` | images/sec | Throughput |
| `time_sec` | seconds | Total inference time |
| `avg_batch_time_sec` | seconds | Mean time per batch |
| `count` | int | Images processed |
| `embedding_shape` | list | Output dimension (e.g. `[1, 768]`) |

## Usage

```bash
# Start server with multimodal enabled
cd scripts/embedding/sglang
SGLANG_ENABLE_MULTIMODAL=1 MODEL_DIR=/path/to/model ./start_sglang_server.sh

# Run image embedding benchmark
cd scripts/vl-embedding
IMAGE_DIR=./images IMAGE_SIZE=512x512 BATCH_SIZE=16 ./run_fix_image_size.sh
```

## Scale Testing

Scale-test support via `scripts/scale-test/vl-embedding/`:
- `run_scale_fix_image_size.py` — shim to the generic scale-test runner
- `gen_test_images.py` — generate synthetic test images (checkerboard/gradient/noise)
- Configs in `config/local/` and `config/t-cloud/`

See [Remote Deployment Guide](../../guides/remote-deployment.md) for multi-host execution.

## Related

- [Embedding Task](embedding.md) — text embedding
- [SGLang Backend](../backends/sglang.md) — server configuration
- [vLLM Backend](../backends/vllm.md) — server configuration
- [Batch Size Tuning](../../concepts/batch-size-tuning.md)
