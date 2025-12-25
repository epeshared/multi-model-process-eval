#!/usr/bin/env bash
set -euo pipefail

# Run Qwen2.5-VL on Flickr8k images via scripts/py/run_vl.py.
#
# Env overrides:
#   MODEL (default: qwen2.5-vl-7b-instruct)
#   MODEL_ID (default: Qwen/Qwen2.5-VL-7B-Instruct)
#   BACKEND (default: torch)
#   PROMPT (default: "Describe the image.")
#   FLICKR8K_IMAGES_DIR
#   FLICKR8K_CAPTIONS_FILE (Flickr8k.token.txt)
#   MAX_SAMPLES
#   BATCH_SIZE
#   DEVICE
#   DTYPE
#   USE_AMX (true/1/yes/on to enable; torch+cpu only)
#   PRINT_MODEL_INFO (true/1/yes/on to enable; prints model/client info at load)
#   WARMUP (int; number of warmup calls after loading session; excluded from timing)
#
# HTTP backends:
#   BASE_URL (for BACKEND=sglang or BACKEND=vllm-http)
#   API (default: v1)
#   API_KEY
#   TIMEOUT
#   IMAGE_TRANSPORT (data-url|path/url)
#
# Offline backends:
#   TP_SIZE
#   DP_SIZE (sglang-offline)
#   MAX_MODEL_LEN (vllm offline)
#   GPU_MEMORY_UTILIZATION (vllm offline)
#
# Profiling:
#   PROFILE (0/1/true/false)  -> --profile
#   PROFILE_RECORD_SHAPES (0/1) -> --profile-record-shapes
#   PROFILE_ACTIVITIES (default: CPU,CUDA) -> --profile-activities
#   PROFILE_OUT_DIR (default: "") -> --profile-out-dir
#   PROFILE_OUT_NAME (default: vl_profile) -> --profile-out-name
#   PROFILE_STRICT (0/1) -> --profile-strict

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL=${MODEL:-qwen2.5-vl-7b-instruct}
MODEL_ID=${MODEL_ID:-/mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-VL-7B-Instruct}
BACKEND=${BACKEND:-torch}
PROMPT=${PROMPT:-"Describe the image."}

BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
API=${API:-v1}
API_KEY=${API_KEY:-}
TIMEOUT=${TIMEOUT:-600}
IMAGE_TRANSPORT=${IMAGE_TRANSPORT:-data-url}

TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

FLICKR8K_IMAGES_DIR=${FLICKR8K_IMAGES_DIR:-/home/xtang/datasets/Flickr8k/Flicker8k_Dataset}
FLICKR8K_CAPTIONS_FILE=${FLICKR8K_CAPTIONS_FILE:-/home/xtang/datasets/Flickr8k/Flickr8k.token.txt}

MAX_SAMPLES=${MAX_SAMPLES:-50}
BATCH_SIZE=${BATCH_SIZE:-1}
# DEVICE=${DEVICE:-cuda:0}
DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-auto}
USE_AMX=${USE_AMX:-0}
PRINT_MODEL_INFO=${PRINT_MODEL_INFO:-0}
WARMUP=${WARMUP:-0}

# Profile envs
PROFILE=${PROFILE:-0}
PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-true}
PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES:-CPU,CUDA}
PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-sglang_logs/sglang_$PROFILE_ACTIVITIES}
PROFILE_OUT_NAME=${PROFILE_OUT_NAME:-vl_profile}
PROFILE_STRICT=${PROFILE_STRICT:-0}
PROFILE_RECORD_SHAPE=${PROFILE_RECORD_SHAPE:-true}

# Optional positional overrides:
#   $1 -> captions file
#   $2 -> images dir
if [[ $# -gt 0 ]]; then
  FLICKR8K_CAPTIONS_FILE="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  FLICKR8K_IMAGES_DIR="$1"
  shift
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
# Profiling args builder
# -------------------------
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

    # Only meaningful for offline (torch.profiler export), but safe to pass always.
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

# Align env vars with scripts/embedding/sglang/start_sglang_server.sh when using sglang-offline.
if [[ "${BACKEND}" == "sglang-offline" || "${BACKEND}" == "sglang_offline" ]]; then
  export DNNL_MAX_CPU_ISA="${DNNL_MAX_CPU_ISA:-AVX512_CORE_AMX}"
  export DNNL_VERBOSE="${DNNL_VERBOSE:-0}"
  export IPEX_DISABLE_AUTOCAST="${IPEX_DISABLE_AUTOCAST:-1}"

  export SGLANG_USE_CPU_ENGINE="${SGLANG_USE_CPU_ENGINE:-1}"
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-1}"

  # Prefer the active conda env, but fall back to the known env path.
  export CONDA_PREFIX="${CONDA_PREFIX:-/root/miniforge3/envs/xtang-embedding-cpu}"

  # Profiler/log dir.
  _SGLANG_LOG_DIR="${ROOT_DIR}/scripts/vl/sglang/sglang_logs/sglang_cpu"
  mkdir -p "${_SGLANG_LOG_DIR}"
  export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-${_SGLANG_LOG_DIR}}"

  # If user didn't set PROFILE_OUT_DIR, default it to SGLANG_TORCH_PROFILER_DIR when profiling is enabled.
  if [[ -n "${PROFILE_ARGS[*]}" ]] && [[ -z "${PROFILE_OUT_DIR}" ]]; then
    # update PROFILE_ARGS in-place (append out dir)
    PROFILE_ARGS+=(--profile-out-dir "${SGLANG_TORCH_PROFILER_DIR}")
  fi

  # Safe LD_PRELOAD join (only add libs that exist; don't clobber existing preload).
  _existing_preload="${LD_PRELOAD:-}"
  _preload_join=""
  _libs=(
    "${CONDA_PREFIX}/lib/libiomp5.so"
    "${CONDA_PREFIX}/lib/libtcmalloc.so"
    "${CONDA_PREFIX}/lib/libtbbmalloc.so.2"
  )
  for f in "${_libs[@]}"; do
    [[ -f "${f}" ]] && _preload_join="${_preload_join:+${_preload_join}:}${f}"
  done
  if [[ -n "${_preload_join}" ]]; then
    if [[ -n "${_existing_preload}" ]]; then
      export LD_PRELOAD="${_preload_join}:${_existing_preload}"
    else
      export LD_PRELOAD="${_preload_join}"
    fi
  fi
fi

# If user already passed --warmup via EXTRA_ARGS, don't add another one.
WARMUP_ARG=()
if [[ "${WARMUP}" != "" && "${WARMUP}" != "0" ]]; then
  if [[ " $* " != *" --warmup "* ]]; then
    WARMUP_ARG=(--warmup "${WARMUP}")
  fi
fi

cd "${ROOT_DIR}"

echo "[run_qwen_vl_flickr8k] MODEL=${MODEL}"
echo "[run_qwen_vl_flickr8k] MODEL_ID=${MODEL_ID}"
echo "[run_qwen_vl_flickr8k] BACKEND=${BACKEND}"
echo "[run_qwen_vl_flickr8k] BASE_URL=${BASE_URL}"
echo "[run_qwen_vl_flickr8k] API=${API}"
echo "[run_qwen_vl_flickr8k] IMAGE_TRANSPORT=${IMAGE_TRANSPORT}"
echo "[run_qwen_vl_flickr8k] TP_SIZE=${TP_SIZE}"
echo "[run_qwen_vl_flickr8k] DP_SIZE=${DP_SIZE}"
echo "[run_qwen_vl_flickr8k] MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "[run_qwen_vl_flickr8k] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[run_qwen_vl_flickr8k] PROMPT=${PROMPT}"
echo "[run_qwen_vl_flickr8k] FLICKR8K_CAPTIONS_FILE=${FLICKR8K_CAPTIONS_FILE}"
echo "[run_qwen_vl_flickr8k] FLICKR8K_IMAGES_DIR=${FLICKR8K_IMAGES_DIR}"
echo "[run_qwen_vl_flickr8k] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[run_qwen_vl_flickr8k] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_qwen_vl_flickr8k] DEVICE=${DEVICE}"
echo "[run_qwen_vl_flickr8k] DTYPE=${DTYPE}"
echo "[run_qwen_vl_flickr8k] USE_AMX=${USE_AMX}"
echo "[run_qwen_vl_flickr8k] PRINT_MODEL_INFO=${PRINT_MODEL_INFO}"
echo "[run_qwen_vl_flickr8k] WARMUP=${WARMUP}"
echo "[run_qwen_vl_flickr8k] PROFILE=${PROFILE}"
echo "[run_qwen_vl_flickr8k] PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES}"
echo "[run_qwen_vl_flickr8k] PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES}"
echo "[run_qwen_vl_flickr8k] PROFILE_OUT_DIR=${PROFILE_OUT_DIR}"
echo "[run_qwen_vl_flickr8k] PROFILE_OUT_NAME=${PROFILE_OUT_NAME}"
echo "[run_qwen_vl_flickr8k] PROFILE_STRICT=${PROFILE_STRICT}"

if [[ ${#PROFILE_ARGS[@]} -gt 0 ]]; then
  printf '[run_qwen_vl_flickr8k] PROFILE_ARGS='; printf '%q ' "${PROFILE_ARGS[@]}"; printf '\n'
else
  echo "[run_qwen_vl_flickr8k] PROFILE_ARGS=<none>"
fi

if [[ $# -gt 0 ]]; then
  printf '[run_qwen_vl_flickr8k] EXTRA_ARGS='; printf '%q ' "$@"; printf '\n'
else
  echo "[run_qwen_vl_flickr8k] EXTRA_ARGS=<none>"
fi

python scripts/py/run_vl.py \
  --model "${MODEL}" \
  --model-id "${MODEL_ID}" \
  --backend "${BACKEND}" \
  --dataset flickr8k \
  --dataset-path "${FLICKR8K_CAPTIONS_FILE}" \
  --flickr8k-captions-file "${FLICKR8K_CAPTIONS_FILE}" \
  --flickr8k-images-dir "${FLICKR8K_IMAGES_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --prompt "${PROMPT}" \
  "${WARMUP_ARG[@]}" \
  "${PROFILE_ARGS[@]}" \
  ${DEVICE:+--device "${DEVICE}"} \
  ${DTYPE:+--dtype "${DTYPE}"} \
  ${BASE_URL:+--base-url "${BASE_URL}"} \
  --api "${API}" \
  --api-key "${API_KEY}" \
  --timeout "${TIMEOUT}" \
  --image-transport "${IMAGE_TRANSPORT}" \
  --tp-size "${TP_SIZE}" \
  --dp-size "${DP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "${USE_AMX_ARG[@]}" \
  "${PRINT_MODEL_INFO_ARG[@]}" \
  "$@"
