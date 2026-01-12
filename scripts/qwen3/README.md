# Qwen3 test scripts

- `run_qwen3_test.sh`: Synthetic text stress-test for Qwen3 via `scripts/qwen3/run_qwen3.py`.

## Start servers

### vLLM (OpenAI-compatible)
```bash
cd scripts/qwen3/vllm
MODEL_DIR=/path/to/Qwen3-0.6B SERVED_MODEL_NAME=qwen3-0.6b ./start_vllm_server.sh
```

### SGLang
```bash
cd scripts/qwen3/sglang
MODEL_DIR=/path/to/Qwen3-0.6B ./start_sglang_server.sh
# CUDA:
MODEL_DIR=/path/to/Qwen3-1.7B ./start_sglang_server_cuda.sh
```

## Usage
```bash
# vLLM HTTP (default)
MODEL=qwen3-0.6b MODEL_ID=/mnt/models/Qwen/Qwen3-0.6B BASE_URL=http://127.0.0.1:8000 ./run_qwen3_test.sh

# SGLang HTTP
MODEL=qwen3-1.7b MODEL_ID=/mnt/models/Qwen/Qwen3-1.7B BACKEND=sglang BASE_URL=http://127.0.0.1:30000 ./run_qwen3_test.sh

# Tune synthetic dataset
SYNTHETIC_NUM_PROMPTS=50 SYNTHETIC_TOKEN_LEN=64 BATCH_SIZE=2 MAX_NEW_TOKENS=128 ./run_qwen3_test.sh
```
