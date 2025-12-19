#!/usr/bin/env bash
set -euo pipefail

# Run embeddings on Flickr8k (images + captions) via scripts/py/run_embedding.py.
#
# Environment overrides (aligned with embedding_eval/scripts/bench_flickr8k_sglang.sh style):
#   MODEL (default: clip-vit-base-patch32)
#   MODEL_ID (default: /home/xtang/models/openai/clip-vit-base-patch32)
#   BACKEND (default: torch)                   # torch|sglang|sglang-offline|vllm|vllm-http
#   BASE_URL (default: http://127.0.0.1:30000) # for BACKEND=sglang
#   API (default: v1)                          # for BACKEND=sglang (v1 recommended for images)
#   API_KEY (default: empty)
#   IMAGE_TRANSPORT (default: data-url)
#
# Flickr8k dataset:
#   FLICKR8K_IMAGES_DIR
#   FLICKR8K_CAPTIONS_FILE
#   FLICKR8K_MODALITY (both|text|image)
#   CAPTIONS_PER_IMAGE
#
# Perf:
#   MAX_SAMPLES (max images; -1 for all)
#   BATCH_SIZE
#   DEVICE
#   DTYPE
#   WARMUP_SAMPLES (if >1, do warmup)
#   USE_AMX (true/1/yes/on to enable; torch+cpu only)

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MODEL=${MODEL:-clip-vit-base-patch32}
MODEL_ID=${MODEL_ID:-/home/xtang/models/openai/clip-vit-base-patch32}
BACKEND=${BACKEND:-torch}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
API=${API:-v1}
API_KEY=${API_KEY:-}
IMAGE_TRANSPORT=${IMAGE_TRANSPORT:-data-url}

FLICKR8K_IMAGES_DIR=${FLICKR8K_IMAGES_DIR:-/home/xtang/datasets/Flickr8k/Flicker8k_Dataset}
FLICKR8K_CAPTIONS_FILE=${FLICKR8K_CAPTIONS_FILE:-/home/xtang/datasets/Flickr8k/Flickr8k.token.txt}
FLICKR8K_MODALITY=${FLICKR8K_MODALITY:-both}
CAPTIONS_PER_IMAGE=${CAPTIONS_PER_IMAGE:-1}

MAX_SAMPLES=${MAX_SAMPLES:-1000}
BATCH_SIZE=${BATCH_SIZE:-100}
DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-bfloat16}
WARMUP_SAMPLES=${WARMUP_SAMPLES:-1000}
USE_AMX=${USE_AMX:-0}

# Optional positional overrides:
#   $1 -> captions file (Flickr8k.token.txt)
#   $2 -> images dir
if [[ $# -gt 0 ]]; then
  FLICKR8K_CAPTIONS_FILE="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  FLICKR8K_IMAGES_DIR="$1"
  shift
fi

if [[ -z "${FLICKR8K_CAPTIONS_FILE}" ]]; then
  echo "Usage: $0 <Flickr8k.token.txt> [Flicker8k_Dataset_dir] [extra python args...]" >&2
  echo "Hint: set FLICKR8K_CAPTIONS_FILE env var or pass it as the first arg." >&2
  exit 1
fi

cd "${ROOT_DIR}"

echo "[run_embedding_flickr8k] MODEL=${MODEL}"
echo "[run_embedding_flickr8k] MODEL_ID=${MODEL_ID:-<unset>}"
echo "[run_embedding_flickr8k] BACKEND=${BACKEND}"
echo "[run_embedding_flickr8k] BASE_URL=${BASE_URL:-<unset>}"
echo "[run_embedding_flickr8k] API=${API}"
echo "[run_embedding_flickr8k] IMAGE_TRANSPORT=${IMAGE_TRANSPORT}"
echo "[run_embedding_flickr8k] FLICKR8K_CAPTIONS_FILE=${FLICKR8K_CAPTIONS_FILE}"
echo "[run_embedding_flickr8k] FLICKR8K_IMAGES_DIR=${FLICKR8K_IMAGES_DIR:-<unset>}"
echo "[run_embedding_flickr8k] FLICKR8K_MODALITY=${FLICKR8K_MODALITY}"
echo "[run_embedding_flickr8k] CAPTIONS_PER_IMAGE=${CAPTIONS_PER_IMAGE}"
echo "[run_embedding_flickr8k] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[run_embedding_flickr8k] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_embedding_flickr8k] DEVICE=${DEVICE:-<unset>}"
echo "[run_embedding_flickr8k] DTYPE=${DTYPE:-<unset>}"
echo "[run_embedding_flickr8k] WARMUP_SAMPLES=${WARMUP_SAMPLES}"
echo "[run_embedding_flickr8k] USE_AMX=${USE_AMX}"
if [[ $# -gt 0 ]]; then
  printf '[run_embedding_flickr8k] EXTRA_ARGS='; printf '%q ' "$@"; printf '\n'
else
  echo "[run_embedding_flickr8k] EXTRA_ARGS=<none>"
fi

MODEL_ID_ARG=()
if [[ -n "${MODEL_ID:-}" ]]; then
  MODEL_ID_ARG=(--model-id "${MODEL_ID}")
fi

USE_AMX_ARG=()
case "${USE_AMX}" in
  1|true|TRUE|yes|YES|on|ON)
    USE_AMX_ARG=(--use-amx)
    ;;
esac

python scripts/py/run_embedding.py \
  --model "${MODEL}" \
  "${MODEL_ID_ARG[@]}" \
  --backend "${BACKEND}" \
  --dataset flickr8k \
  --dataset-path "${FLICKR8K_CAPTIONS_FILE}" \
  --flickr8k-captions-file "${FLICKR8K_CAPTIONS_FILE}" \
  ${FLICKR8K_IMAGES_DIR:+--flickr8k-images-dir "${FLICKR8K_IMAGES_DIR}"} \
  --flickr8k-modality "${FLICKR8K_MODALITY}" \
  --flickr8k-captions-per-image "${CAPTIONS_PER_IMAGE}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup-samples "${WARMUP_SAMPLES}" \
  "${USE_AMX_ARG[@]}" \
  ${BASE_URL:+--base-url "${BASE_URL}"} \
  --api "${API}" \
  --api-key "${API_KEY}" \
  --image-transport "${IMAGE_TRANSPORT}" \
  ${DEVICE:+--device "${DEVICE}"} \
  ${DTYPE:+--dtype "${DTYPE}"} \
  "$@"
