#!/usr/bin/env bash
set -euo pipefail

# Synthetic text stress-test for Qwen3 via scripts/qwen3/run_qwen3.py.
#
# Env overrides:
#   MODEL (default: qwen3-0.6b)
#   MODEL_ID (default: /mnt/models/Qwen/Qwen3-0.6B)
#   BACKEND (default: vllm-http)   # sglang | vllm-http
#   PROMPT (default: "Write a short sentence.")
#
# Synthetic dataset:
#   SYNTHETIC_NUM_PROMPTS (default: 10)
#   SYNTHETIC_TOKEN_LEN (default: 32)
#   SYNTHETIC_SEED (default: 1234)
#   BATCH_SIZE (default: 1)
#
# Runtime:
#   MAX_NEW_TOKENS (default: 128)
#   WARMUP (default: 1)
#
# HTTP backends:
#   BASE_URL (default depends on backend)
#   API_KEY
#   TIMEOUT

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL=${MODEL:-qwen3-0.6b}
MODEL_ID=${MODEL_ID:-/mnt/models/Qwen/Qwen3-0.6B}
BACKEND=${BACKEND:-vllm-http}
PROMPT=${PROMPT:-"Write a short sentence."}

SYNTHETIC_NUM_PROMPTS=${SYNTHETIC_NUM_PROMPTS:-10}
SYNTHETIC_TOKEN_LEN=${SYNTHETIC_TOKEN_LEN:-32}
SYNTHETIC_SEED=${SYNTHETIC_SEED:-1234}
BATCH_SIZE=${BATCH_SIZE:-1}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
WARMUP=${WARMUP:-1}

API_KEY=${API_KEY:-}
TIMEOUT=${TIMEOUT:-600}

if [[ -z "${BASE_URL:-}" ]]; then
	# defaults per backend
	if [[ "${BACKEND}" == "sglang" ]]; then
		BASE_URL=http://127.0.0.1:30000
	else
		BASE_URL=http://127.0.0.1:8000
	fi
fi

cd "${ROOT_DIR}"

echo "[run_qwen3_test] MODEL=${MODEL}"
echo "[run_qwen3_test] MODEL_ID=${MODEL_ID}"
echo "[run_qwen3_test] BACKEND=${BACKEND}"
echo "[run_qwen3_test] BASE_URL=${BASE_URL}"
echo "[run_qwen3_test] SYNTHETIC_NUM_PROMPTS=${SYNTHETIC_NUM_PROMPTS}"
echo "[run_qwen3_test] SYNTHETIC_TOKEN_LEN=${SYNTHETIC_TOKEN_LEN}"
echo "[run_qwen3_test] SYNTHETIC_SEED=${SYNTHETIC_SEED}"
echo "[run_qwen3_test] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_qwen3_test] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[run_qwen3_test] WARMUP=${WARMUP}"

python scripts/qwen3/run_qwen3.py \
	--model "${MODEL}" \
	--model-id "${MODEL_ID}" \
	--backend "${BACKEND}" \
	--dataset synthetic \
	--prompt "${PROMPT}" \
	--synthetic-num-prompts "${SYNTHETIC_NUM_PROMPTS}" \
	--synthetic-token-len "${SYNTHETIC_TOKEN_LEN}" \
	--synthetic-seed "${SYNTHETIC_SEED}" \
	--batch-size "${BATCH_SIZE}" \
	--max-new-tokens "${MAX_NEW_TOKENS}" \
	--warmup "${WARMUP}" \
	--base-url "${BASE_URL}" \
	--api-key "${API_KEY}" \
	--timeout "${TIMEOUT}"
