#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODEL_DIR=/path/to/Qwen3 ./run_qwen3_test.sh MODEL_SIZE
# Example:
#   MODEL_DIR=/mnt/models/Qwen/Qwen3-0.6B ./run_qwen3_test.sh 0.6B

MODEL_SIZE=${1:-"0.6B"}
MODEL_DIR=${MODEL_DIR:-"/mnt/models/Qwen/Qwen3-${MODEL_SIZE}"}
BACKEND=${BACKEND:-vllm-http}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}

python ../../src/tasks/qwen3.py --model "$MODEL_DIR" --backend "$BACKEND" --base-url "$BASE_URL" --test
