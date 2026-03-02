#!/usr/bin/env bash
set -euo pipefail

# Benchmark image embeddings against an SGLang embedding server (multimodal).
#
# Env:
#   MODEL (logical key; informational)
#   MODEL_ID (served model id string; default: Qwen3-VL-Embedding-2B)
#   BACKEND (default: sglang)
#   BASE_URL (default: http://127.0.0.1:30000)
#   HOST/PORT (used by server start script; informational here)
#   MAX_SAMPLES (default: 1000)
#   BATCH_SIZE (default: 32)
#   IMAGE_DIR (directory containing images)
#   IMAGE_SIZE (optional filter tag like 512x512; matched in filename)
#   IMAGE_TRANSPORT (data-url|base64|path/url; default: data-url)
#   EMBEDDING_HTTP_TIMEOUT (seconds; default: 900)
#   WARMUP_SAMPLES (default: 1)
#
# NOTE: For multimodal models, the SGLang server must be launched with multimodal enabled.
#       (e.g. start_sglang_server_cuda.sh already passes --enable-multimodal)

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL=${MODEL:-qwen3-vl-embedding-2b}
MODEL_ID=${MODEL_ID:-Qwen3-VL-Embedding-2B}
BACKEND=${BACKEND:-sglang}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}

MAX_SAMPLES=${MAX_SAMPLES:-1000}
BATCH_SIZE=${BATCH_SIZE:-32}
WARMUP_SAMPLES=${WARMUP_SAMPLES:-1}

IMAGE_DIR=${IMAGE_DIR:-}
IMAGE_SIZE=${IMAGE_SIZE:-}
IMAGE_TRANSPORT=${IMAGE_TRANSPORT:-data-url}
EMBEDDING_HTTP_TIMEOUT=${EMBEDDING_HTTP_TIMEOUT:-900}

# Choose Python interpreter for the client benchmark.
# Note: some configs set VLLM_CONDA_ENV (for the server). If the local runner
# environment does not have torch installed, we also need the client to run
# inside that env. We support this without hard-coding an absolute python path.
EMBEDDING_PYTHON_CMD=()
if [[ -n "${EMBEDDING_PYTHON:-}" ]]; then
  EMBEDDING_PYTHON_CMD=("${EMBEDDING_PYTHON}")
elif [[ -n "${SGLANG_PYTHON:-}" ]]; then
  EMBEDDING_PYTHON_CMD=("${SGLANG_PYTHON}")
elif [[ -n "${VLLM_PYTHON:-}" ]]; then
  EMBEDDING_PYTHON_CMD=("${VLLM_PYTHON}")
elif [[ -n "${VLLM_CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found but VLLM_CONDA_ENV=${VLLM_CONDA_ENV} is set" >&2
    exit 2
  fi
  EMBEDDING_PYTHON_CMD=(conda run -n "${VLLM_CONDA_ENV}" python)
else
  EMBEDDING_PYTHON_CMD=(python)
fi

cd "${ROOT_DIR}"

echo "[run_fix_image_size] MODEL=${MODEL}"
echo "[run_fix_image_size] MODEL_ID=${MODEL_ID}"
echo "[run_fix_image_size] BACKEND=${BACKEND}"
echo "[run_fix_image_size] BASE_URL=${BASE_URL}"
echo "[run_fix_image_size] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[run_fix_image_size] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_fix_image_size] WARMUP_SAMPLES=${WARMUP_SAMPLES}"
echo "[run_fix_image_size] IMAGE_DIR=${IMAGE_DIR:-<unset>}"
echo "[run_fix_image_size] IMAGE_SIZE=${IMAGE_SIZE:-<unset>}"
echo "[run_fix_image_size] IMAGE_TRANSPORT=${IMAGE_TRANSPORT}"
echo "[run_fix_image_size] EMBEDDING_HTTP_TIMEOUT=${EMBEDDING_HTTP_TIMEOUT}"

if [[ -z "${IMAGE_DIR}" ]]; then
  # Fallback: use a per-run cache folder inside this repo.
  # We do NOT auto-generate here to avoid skewing measured embedding time.
  echo "Error: IMAGE_DIR is required (path to local images)." >&2
  exit 2
fi

"${EMBEDDING_PYTHON_CMD[@]}" "${SCRIPT_DIR}/run_image_embedding.py" \
  --backend "${BACKEND}" \
  --model-id "${MODEL_ID}" \
  --base-url "${BASE_URL}" \
  --api v1 \
  --timeout "${EMBEDDING_HTTP_TIMEOUT}" \
  --image-transport "${IMAGE_TRANSPORT}" \
  --images-dir "${IMAGE_DIR}" \
  ${IMAGE_SIZE:+--image-size "${IMAGE_SIZE}"} \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup-samples "${WARMUP_SAMPLES}"
