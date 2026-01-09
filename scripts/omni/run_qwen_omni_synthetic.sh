#!/usr/bin/env bash
set -euo pipefail

# Stress-test Qwen2.5-Omni on synthetic random images via scripts/omni/run_omni.py.
#
# Env overrides:
#   MODEL (default: qwen2.5-omni-7b)
#   MODEL_ID (default: /mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-Omni-7B)
#   BACKEND (default: sglang)   # sglang | vllm | vllm-http
#   PROMPT (default: "Describe the image.")
#
# Synthetic dataset:
#   SYNTHETIC_IMAGE_SIZE (default: 224x224)
#   SYNTHETIC_NUM_IMAGES (default: 50)
#   SYNTHETIC_SEED (default: 1234)
#   SYNTHETIC_OUT_DIR (default: "")  # empty => temp dir created by run_omni.py
#   BATCH_SIZE (default: 1)  # group images per call (best-effort; backend-dependent)
#
# Runtime:
#   MAX_NEW_TOKENS (default: 128)
#   WARMUP (int; number of warmup calls after loading session)
#
# HTTP backends:
#   BASE_URL (for BACKEND=sglang or BACKEND=vllm-http)
#   API_KEY
#   TIMEOUT
#   IMAGE_TRANSPORT (data-url|path/url)
#
# Offline vLLM:
#   TP_SIZE
#   MAX_MODEL_LEN
#   GPU_MEMORY_UTILIZATION
#   DEVICE
#   DTYPE
#
# Profiling (sglang-http only):
#   PROFILE (0/1/true/false)
#   PROFILE_RECORD_SHAPES (0/1)
#   PROFILE_ACTIVITIES (default: CPU,CUDA)
#   PROFILE_OUT_DIR
#   PROFILE_OUT_NAME (default: omni_profile)
#   PROFILE_STRICT (0/1)
#
# Optional positional overrides:
#   $1 -> SYNTHETIC_IMAGE_SIZE
#   $2 -> SYNTHETIC_NUM_IMAGES
#   $3 -> SYNTHETIC_OUT_DIR

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL=${MODEL:-qwen2.5-omni-3b}
MODEL_ID=${MODEL_ID:-/mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-Omni-3B}
BACKEND=${BACKEND:-sglang}
PROMPT=${PROMPT:-"Describe the image."}

BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
API_KEY=${API_KEY:-}
TIMEOUT=${TIMEOUT:-600}
IMAGE_TRANSPORT=${IMAGE_TRANSPORT:-data-url}

TP_SIZE=${TP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

DEVICE=${DEVICE:-}
DTYPE=${DTYPE:-auto}

SYNTHETIC_IMAGE_SIZE=${SYNTHETIC_IMAGE_SIZE:-224x224}
SYNTHETIC_NUM_IMAGES=${SYNTHETIC_NUM_IMAGES:-10}
SYNTHETIC_SEED=${SYNTHETIC_SEED:-1234}
SYNTHETIC_OUT_DIR=${SYNTHETIC_OUT_DIR:-}

BATCH_SIZE=${BATCH_SIZE:-1}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
WARMUP=${WARMUP:-1}

PROFILE=${PROFILE:-0}
PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-true}
PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES:-CPU,CUDA}
PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-}
PROFILE_OUT_NAME=${PROFILE_OUT_NAME:-omni_profile}
PROFILE_STRICT=${PROFILE_STRICT:-0}

# Optional positional overrides
if [[ $# -gt 0 ]]; then
  SYNTHETIC_IMAGE_SIZE="$1"; shift
fi
if [[ $# -gt 0 ]]; then
  SYNTHETIC_NUM_IMAGES="$1"; shift
fi
if [[ $# -gt 0 ]]; then
  SYNTHETIC_OUT_DIR="$1"; shift
fi

PROFILE_ARGS=()
case "${PROFILE}" in
  1|true|TRUE|yes|YES|on|ON)
    PROFILE_ARGS+=(--profile)

    case "${PROFILE_RECORD_SHAPES}" in
      1|true|TRUE|yes|YES|on|ON)
        PROFILE_ARGS+=(--profile-record-shapes)
        ;;
    esac

    if [[ -n "${PROFILE_ACTIVITIES}" ]]; then
      PROFILE_ARGS+=(--profile-activities "${PROFILE_ACTIVITIES}")
    fi

    if [[ -n "${PROFILE_OUT_DIR}" ]]; then
      PROFILE_ARGS+=(--profile-out-dir "${PROFILE_OUT_DIR}")
    fi

    if [[ -n "${PROFILE_OUT_NAME}" ]]; then
      PROFILE_ARGS+=(--profile-out-name "${PROFILE_OUT_NAME}")
    fi

    case "${PROFILE_STRICT}" in
      1|true|TRUE|yes|YES|on|ON)
        PROFILE_ARGS+=(--profile-strict)
        ;;
    esac
    ;;
esac

WARMUP_ARG=()
if [[ "${WARMUP}" != "" && "${WARMUP}" != "0" ]]; then
  if [[ " $* " != *" --warmup "* ]]; then
    WARMUP_ARG=(--warmup "${WARMUP}")
  fi
fi

SYNTHETIC_OUT_DIR_ARG=()
if [[ -n "${SYNTHETIC_OUT_DIR}" ]]; then
  SYNTHETIC_OUT_DIR_ARG=(--synthetic-out-dir "${SYNTHETIC_OUT_DIR}")
fi

cd "${ROOT_DIR}"

echo "[run_qwen_omni_synthetic] MODEL=${MODEL}"
echo "[run_qwen_omni_synthetic] MODEL_ID=${MODEL_ID}"
echo "[run_qwen_omni_synthetic] BACKEND=${BACKEND}"
echo "[run_qwen_omni_synthetic] BASE_URL=${BASE_URL}"
echo "[run_qwen_omni_synthetic] IMAGE_TRANSPORT=${IMAGE_TRANSPORT}"
echo "[run_qwen_omni_synthetic] TP_SIZE=${TP_SIZE}"
echo "[run_qwen_omni_synthetic] MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "[run_qwen_omni_synthetic] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[run_qwen_omni_synthetic] PROMPT=${PROMPT}"
echo "[run_qwen_omni_synthetic] SYNTHETIC_IMAGE_SIZE=${SYNTHETIC_IMAGE_SIZE}"
echo "[run_qwen_omni_synthetic] SYNTHETIC_NUM_IMAGES=${SYNTHETIC_NUM_IMAGES}"
echo "[run_qwen_omni_synthetic] SYNTHETIC_SEED=${SYNTHETIC_SEED}"
echo "[run_qwen_omni_synthetic] SYNTHETIC_OUT_DIR=${SYNTHETIC_OUT_DIR:-<tmp>}"
echo "[run_qwen_omni_synthetic] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_qwen_omni_synthetic] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[run_qwen_omni_synthetic] WARMUP=${WARMUP}"
echo "[run_qwen_omni_synthetic] PROFILE=${PROFILE}"

if [[ ${#PROFILE_ARGS[@]} -gt 0 ]]; then
  printf '[run_qwen_omni_synthetic] PROFILE_ARGS='; printf '%q ' "${PROFILE_ARGS[@]}"; printf '\n'
else
  echo "[run_qwen_omni_synthetic] PROFILE_ARGS=<none>"
fi

if [[ $# -gt 0 ]]; then
  printf '[run_qwen_omni_synthetic] EXTRA_ARGS='; printf '%q ' "$@"; printf '\n'
else
  echo "[run_qwen_omni_synthetic] EXTRA_ARGS=<none>"
fi

python scripts/omni/run_omni.py \
  --model "${MODEL}" \
  --model-id "${MODEL_ID}" \
  --backend "${BACKEND}" \
  --dataset synthetic \
  --synthetic-image-size "${SYNTHETIC_IMAGE_SIZE}" \
  --synthetic-num-images "${SYNTHETIC_NUM_IMAGES}" \
  --synthetic-seed "${SYNTHETIC_SEED}" \
  --batch-size "${BATCH_SIZE}" \
  "${SYNTHETIC_OUT_DIR_ARG[@]}" \
  --prompt "${PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${WARMUP_ARG[@]}" \
  "${PROFILE_ARGS[@]}" \
  ${BASE_URL:+--base-url "${BASE_URL}"} \
  --api-key "${API_KEY}" \
  --timeout "${TIMEOUT}" \
  --image-transport "${IMAGE_TRANSPORT}" \
  --tp-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  ${DEVICE:+--device "${DEVICE}"} \
  ${DTYPE:+--dtype "${DTYPE}"} \
  "$@"
