#!/usr/bin/env bash
set -euo pipefail

# Run embeddings on Flickr8k (images + captions) via scripts/py/run_embedding.py.
#
# Environment overrides:
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
#   PRINT_MODEL_INFO (true/1/yes/on to enable)
#
# Profiling (new):
#   PROFILE (true/1/yes/on)                    # enables --profile
#   PROFILE_RECORD_SHAPES (true/1/yes/on)      # enables --profile-record-shapes
#   PROFILE_ACTIVITIES (default: CPU,CUDA)     # e.g. CPU,CUDA
#   PROFILE_OUT_DIR                            # --profile-out-dir
#   PROFILE_OUT_NAME (default: embedding_profile)
#   PROFILE_STRICT (true/1/yes/on)             # --profile-strict

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

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
PRINT_MODEL_INFO=${PRINT_MODEL_INFO:-0}

# Profiling envs
PROFILE=${PROFILE:-0}
PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-0}
PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES:-CPU,CUDA}
PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-}
PROFILE_OUT_NAME=${PROFILE_OUT_NAME:-embedding_profile}
PROFILE_STRICT=${PROFILE_STRICT:-0}

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
echo "[run_embedding_flickr8k] PRINT_MODEL_INFO=${PRINT_MODEL_INFO}"
echo "[run_embedding_flickr8k] PROFILE=${PROFILE}"
echo "[run_embedding_flickr8k] PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES}"
echo "[run_embedding_flickr8k] PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES}"
echo "[run_embedding_flickr8k] PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-<unset>}"
echo "[run_embedding_flickr8k] PROFILE_OUT_NAME=${PROFILE_OUT_NAME}"
echo "[run_embedding_flickr8k] PROFILE_STRICT=${PROFILE_STRICT}"
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

PRINT_MODEL_INFO_ARG=()
case "${PRINT_MODEL_INFO}" in
  1|true|TRUE|yes|YES|on|ON)
    PRINT_MODEL_INFO_ARG=(--print-model-info)
    ;;
esac

# -------------------------
# Profiling args (new)
# - If user already passed the flag in EXTRA_ARGS, don't add duplicates.
# -------------------------
PROFILE_ARGS=()

case "${PROFILE}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ " $* " != *" --profile "* ]]; then
      PROFILE_ARGS+=(--profile)
    fi
    ;;
esac

case "${PROFILE_RECORD_SHAPES}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ " $* " != *" --profile-record-shapes "* ]]; then
      PROFILE_ARGS+=(--profile-record-shapes)
    fi
    ;;
esac

if [[ -n "${PROFILE_ACTIVITIES}" ]]; then
  if [[ " $* " != *" --profile-activities "* ]]; then
    PROFILE_ARGS+=(--profile-activities "${PROFILE_ACTIVITIES}")
  fi
fi

if [[ -n "${PROFILE_OUT_DIR}" ]]; then
  if [[ " $* " != *" --profile-out-dir "* ]]; then
    PROFILE_ARGS+=(--profile-out-dir "${PROFILE_OUT_DIR}")
  fi
fi

if [[ -n "${PROFILE_OUT_NAME}" ]]; then
  if [[ " $* " != *" --profile-out-name "* ]]; then
    PROFILE_ARGS+=(--profile-out-name "${PROFILE_OUT_NAME}")
  fi
fi

case "${PROFILE_STRICT}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ " $* " != *" --profile-strict "* ]]; then
      PROFILE_ARGS+=(--profile-strict)
    fi
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
  "${PRINT_MODEL_INFO_ARG[@]}" \
  "${PROFILE_ARGS[@]}" \
  ${BASE_URL:+--base-url "${BASE_URL}"} \
  --api "${API}" \
  --api-key "${API_KEY}" \
  --image-transport "${IMAGE_TRANSPORT}" \
  ${DEVICE:+--device "${DEVICE}"} \
  ${DTYPE:+--dtype "${DTYPE}"} \
  "$@"
