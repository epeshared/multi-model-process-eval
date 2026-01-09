# Omni scripts

This folder contains helper scripts to run and stress-test Omni models via [scripts/omni/run_omni.py](scripts/omni/run_omni.py).

- `run_qwen_omni_synthetic.sh`: generate random images and run Omni chat for throughput testing.
- `sglang/start_sglang_server*.sh`: start an SGLang server for Omni models.
- `vllm/start_vllm_server.sh`: start a vLLM OpenAI-compatible server for Omni models.

## Stress test knobs

`run_qwen_omni_synthetic.sh` supports:

- `SYNTHETIC_NUM_IMAGES`: total number of images.
- `BATCH_SIZE`: groups images per call to the Python runner (best-effort; actual backend batching support varies).
- `MAX_NEW_TOKENS`: generation length.

The JSON output includes `time_sec_per_batch` and `time_sec_per_sample` for easier comparisons.

## Quickstart

### SGLang (HTTP)

1) Start the server:

```bash
cd scripts/omni/sglang
MODEL_DIR=/path/to/Qwen2.5-Omni-7B ./start_sglang_server.sh
```

2) Run the synthetic stress test:

```bash
cd scripts/omni
BACKEND=sglang BASE_URL=http://127.0.0.1:30000 ./run_qwen_omni_synthetic.sh
```

### vLLM (OpenAI-compatible HTTP)

1) Start the server:

```bash
cd scripts/omni/vllm
MODEL_DIR=/path/to/Qwen2.5-Omni-3B SERVED_MODEL_NAME=qwen2.5-omni-3b ./start_vllm_server.sh
# or
MODEL_DIR=/path/to/Qwen2.5-Omni-7B SERVED_MODEL_NAME=qwen2.5-omni-7b ./start_vllm_server.sh
```

2) Run the synthetic stress test:

```bash
cd scripts/omni
BACKEND=vllm-http BASE_URL=http://127.0.0.1:8000 ./run_qwen_omni_synthetic.sh
```

Notes:

- For `BACKEND=vllm-http`, the HTTP request uses the served model name (here we use the CLI `--model` value).
	This must match the `--served-model-name` used when starting vLLM (`SERVED_MODEL_NAME`).
- For `BACKEND=sglang`, the default port is `30000` (matching the SGLang startup script).
