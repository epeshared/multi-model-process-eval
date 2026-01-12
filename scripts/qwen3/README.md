# Qwen3 test scripts

- `run_qwen3_test.sh`: Run Qwen3 model tests for 0.6B, 1.7B, 4B sizes.

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
MODEL_DIR=/mnt/models/Qwen/Qwen3-0.6B ./run_qwen3_test.sh 0.6B
MODEL_DIR=/mnt/models/Qwen/Qwen3-1.7B ./run_qwen3_test.sh 1.7B
MODEL_DIR=/mnt/models/Qwen/Qwen3-4B ./run_qwen3_test.sh 4B
```
