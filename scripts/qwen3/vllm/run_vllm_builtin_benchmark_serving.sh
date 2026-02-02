#!/usr/bin/env bash
set -euo pipefail

# Run vLLM's built-in benchmark_serving against the local vLLM OpenAI-compatible server.
#
# Start server (in another shell):
#   cd scripts/qwen3/vllm
#   MODEL_DIR=/path/to/Qwen3-0.6B SERVED_MODEL_NAME=qwen3-0.6b ./start_vllm_server.sh
#
# Then benchmark:
#   BASE_URL=http://127.0.0.1:8000 MODEL=qwen3-0.6b NUM_PROMPTS=200 CONCURRENCY=16 \
#     IN_LEN=256 OUT_LEN=256 REQUEST_RATE=0 ./run_vllm_builtin_benchmark_serving.sh
#
# You can pass extra args to vLLM benchmark after --, e.g.:
#   ./run_vllm_builtin_benchmark_serving.sh -- --seed 0

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
ENDPOINT=${ENDPOINT:-/v1/chat/completions}
MODEL=${MODEL:-qwen3-0.6b}

NUM_PROMPTS=${NUM_PROMPTS:-200}
CONCURRENCY=${CONCURRENCY:-16}
REQUEST_RATE=${REQUEST_RATE:-0}

IN_LEN=${IN_LEN:-256}
OUT_LEN=${OUT_LEN:-256}
PREFIX_LEN=${PREFIX_LEN:-0}

python "$SCRIPT_DIR/vllm_builtin_benchmark_serving.py" \
  --print-cmd \
  --base-url "$BASE_URL" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --num-prompts "$NUM_PROMPTS" \
  --concurrency "$CONCURRENCY" \
  --request-rate "$REQUEST_RATE" \
  --random-input-len "$IN_LEN" \
  --random-output-len "$OUT_LEN" \
  --random-prefix-len "$PREFIX_LEN" \
  "$@"
