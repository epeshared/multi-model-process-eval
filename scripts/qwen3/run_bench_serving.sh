#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

BACKEND_RAW=${BACKEND:-sglang}
case "${BACKEND_RAW}" in
  sglang)
    BENCH_BACKEND=sglang
    DEFAULT_BASE_URL=http://127.0.0.1:30000
    ;;
  vllm|vllm-http)
    BENCH_BACKEND=vllm
    DEFAULT_BASE_URL=http://127.0.0.1:8000
    ;;
  *)
    BENCH_BACKEND=${BACKEND_RAW}
    DEFAULT_BASE_URL=http://127.0.0.1:30000
    ;;
esac

MODEL=${MODEL:-qwen3-0.6b}
MODEL_PATH=${MODEL_PATH:-${MODEL_DIR:-${MODEL_ID:-Qwen/Qwen3-0.6B}}}
TOKENIZER=${TOKENIZER:-${MODEL_PATH}}
BASE_URL=${BASE_URL:-${DEFAULT_BASE_URL}}
API_KEY=${API_KEY:-}
DATASET_NAME=${DATASET_NAME:-random}

NUM_PROMPTS=${NUM_PROMPTS:-3000}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-${RANDOM_INPUT:-1024}}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-${RANDOM_OUTPUT:-1024}}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
REQUEST_RATE=${REQUEST_RATE:-inf}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-1}
SEED=${SEED:-1}
TOKENIZE_PROMPT=${TOKENIZE_PROMPT:-0}

BENCH_OUTPUT_FILE=${BENCH_OUTPUT_FILE:-${OUTPUT_FILE:-${TMPDIR:-/tmp}/qwen3_bench_${MODEL}_${BENCH_BACKEND}_$$.jsonl}}
BENCH_DISABLE_TQDM=${BENCH_DISABLE_TQDM:-1}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-}

PYTHON_CMD=()
if [[ -n "${BENCH_PYTHON:-}" ]]; then
  PYTHON_CMD=("${BENCH_PYTHON}")
elif [[ -n "${SGLANG_PYTHON:-}" ]]; then
  PYTHON_CMD=("${SGLANG_PYTHON}")
elif [[ -n "${BENCH_CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (needed for BENCH_CONDA_ENV=${BENCH_CONDA_ENV})" >&2
    exit 127
  fi
  PYTHON_CMD=(conda run -n "${BENCH_CONDA_ENV}" python)
elif [[ -n "${SGLANG_CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (needed for SGLANG_CONDA_ENV=${SGLANG_CONDA_ENV})" >&2
    exit 127
  fi
  PYTHON_CMD=(conda run -n "${SGLANG_CONDA_ENV}" python)
else
  PYTHON_BIN="$(command -v python || true)"
  if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
  fi
  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: python not found on PATH" >&2
    exit 127
  fi
  PYTHON_CMD=("${PYTHON_BIN}")
fi

ARGS=(
  -m sglang.bench_serving
  --backend "${BENCH_BACKEND}"
  --dataset-name "${DATASET_NAME}"
  --base-url "${BASE_URL}"
  --model "${MODEL_PATH}"
  --served-model-name "${MODEL}"
  --tokenizer "${TOKENIZER}"
  --num-prompts "${NUM_PROMPTS}"
  --random-input-len "${RANDOM_INPUT_LEN}"
  --random-output-len "${RANDOM_OUTPUT_LEN}"
  --random-range-ratio "${RANDOM_RANGE_RATIO}"
  --request-rate "${REQUEST_RATE}"
  --warmup-requests "${WARMUP_REQUESTS}"
  --seed "${SEED}"
  --output-file "${BENCH_OUTPUT_FILE}"
)

if [[ -n "${MAX_CONCURRENCY}" ]]; then
  ARGS+=(--max-concurrency "${MAX_CONCURRENCY}")
fi
if [[ -n "${API_KEY}" ]]; then
  export OPENAI_API_KEY="${API_KEY}"
fi
if [[ "${TOKENIZE_PROMPT}" == "1" || "${TOKENIZE_PROMPT}" == "true" ]]; then
  ARGS+=(--tokenize-prompt)
fi
if [[ -n "${EXTRA_REQUEST_BODY}" ]]; then
  ARGS+=(--extra-request-body "${EXTRA_REQUEST_BODY}")
fi
if [[ "${BENCH_DISABLE_TQDM}" == "1" || "${BENCH_DISABLE_TQDM}" == "true" ]]; then
  ARGS+=(--disable-tqdm)
fi

cd "${ROOT_DIR}"

echo "[run_bench_serving] BACKEND=${BACKEND_RAW}"
echo "[run_bench_serving] BENCH_BACKEND=${BENCH_BACKEND}"
echo "[run_bench_serving] MODEL=${MODEL}"
echo "[run_bench_serving] MODEL_PATH=${MODEL_PATH}"
echo "[run_bench_serving] BASE_URL=${BASE_URL}"
echo "[run_bench_serving] DATASET_NAME=${DATASET_NAME}"
echo "[run_bench_serving] NUM_PROMPTS=${NUM_PROMPTS}"
echo "[run_bench_serving] RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN}"
echo "[run_bench_serving] RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN}"
echo "[run_bench_serving] RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO}"
echo "[run_bench_serving] REQUEST_RATE=${REQUEST_RATE}"
echo "[run_bench_serving] MAX_CONCURRENCY=${MAX_CONCURRENCY:-<unset>}"
echo "[run_bench_serving] TOKENIZE_PROMPT=${TOKENIZE_PROMPT}"
echo "[run_bench_serving] OUTPUT_FILE=${BENCH_OUTPUT_FILE}"
echo "[run_bench_serving] PYTHON_BIN=${PYTHON_CMD[*]}"

set +e
"${PYTHON_CMD[@]}" "${ARGS[@]}"
rc=$?
set -e

if [[ -f "${BENCH_OUTPUT_FILE}" ]]; then
  echo "[run_bench_serving] OUTPUT_FILE=${BENCH_OUTPUT_FILE}"
  result_json="$(${PYTHON_CMD[@]} - "${BENCH_OUTPUT_FILE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
last = None
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if isinstance(obj, dict):
        last = obj

if last is not None:
    print(json.dumps(last, ensure_ascii=False, sort_keys=True))
PY
)"
  if [[ -n "${result_json}" ]]; then
    echo "[run_bench_serving] RESULT_JSON=${result_json}"
  fi
fi

exit "${rc}"