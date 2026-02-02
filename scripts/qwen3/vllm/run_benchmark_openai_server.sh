#!/usr/bin/env bash
set -euo pipefail

# Wrapper for scripts/qwen3/vllm/benchmark_openai_server.py
#
# Typical usage:
#   # Start server (in another shell)
#   MODEL_DIR=/path/to/Qwen3-0.6B SERVED_MODEL_NAME=qwen3-0.6b ./start_vllm_server.sh
#
#   # Benchmark
#   BASE_URL=http://127.0.0.1:8000 MODEL=qwen3-0.6b NUM_PROMPTS=200 CONCURRENCY=16 MAX_TOKENS=256 ./run_benchmark_openai_server.sh

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
ENDPOINT=${ENDPOINT:-/v1/chat/completions}
MODEL=${MODEL:-qwen3-0.6b}
API_KEY=${API_KEY:-}
TIMEOUT=${TIMEOUT:-600}

NUM_PROMPTS=${NUM_PROMPTS:-200}
CONCURRENCY=${CONCURRENCY:-16}
REQUEST_RATE=${REQUEST_RATE:-0}

PROMPTS_FILE=${PROMPTS_FILE:-}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-256}
SEED=${SEED:-1234}

MAX_TOKENS=${MAX_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0}

STREAM=${STREAM:-1}
SAVE_JSON=${SAVE_JSON:-}

ARGS=(
  --base-url "$BASE_URL"
  --endpoint "$ENDPOINT"
  --model "$MODEL"
  --api-key "$API_KEY"
  --timeout "$TIMEOUT"
  --num-prompts "$NUM_PROMPTS"
  --concurrency "$CONCURRENCY"
  --request-rate "$REQUEST_RATE"
  --random-input-len "$RANDOM_INPUT_LEN"
  --seed "$SEED"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
)

if [[ -n "$PROMPTS_FILE" ]]; then
  ARGS+=(--prompts-file "$PROMPTS_FILE")
fi

if [[ -n "$SAVE_JSON" ]]; then
  ARGS+=(--save-json "$SAVE_JSON")
fi

if [[ "$STREAM" == "0" ]]; then
  ARGS+=(--no-stream)
fi

python "$SCRIPT_DIR/benchmark_openai_server.py" "${ARGS[@]}"
