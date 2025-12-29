#!/usr/bin/env bash
set -euo pipefail

# Run Qwen2.5-VL on synthetic random images via scripts/py/run_vl.py.
#
# Env overrides:
#   MODEL (default: qwen2.5-vl-7b-instruct)
#   MODEL_ID (default: /mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-VL-7B-Instruct)
#   BACKEND (default: torch)
#   PROMPT (default: "Describe the image.")
#
# Synthetic dataset:
#   SYNTHETIC_IMAGE_SIZE (default: 224x224)   # e.g. 224x224, 336,336, "224 224", "224" (square)
#   SYNTHETIC_NUM_IMAGES (default: 50)
#   SYNTHETIC_SEED (default: 1234)
#   SYNTHETIC_OUT_DIR (default: "")           # empty => temp dir created by run_vl.py
#
# Common:
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
#
# Optional positional overrides:
#   $1 -> SYNTHETIC_IMAGE_SIZE
#   $2 -> SYNTHETIC_NUM_IMAGES
#   $3 -> SYNTHETIC_OUT_DIR

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL=${MODEL:-qwen2.5-vl-7b-instruct}
MODEL_ID=${MODEL_ID:-/mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-VL-7B-Instruct}
BACKEND=${BACKEND:-sglang}
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

SYNTHETIC_IMAGE_SIZE=${SYNTHETIC_IMAGE_SIZE:-224x224}
SYNTHETIC_NUM_IMAGES=${SYNTHETIC_NUM_IMAGES:-50}
SYNTHETIC_SEED=${SYNTHETIC_SEED:-1234}
SYNTHETIC_OUT_DIR=${SYNTHETIC_OUT_DIR:-}

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

# Optional positional overrides:
#   $1 -> image size
#   $2 -> num images
#   $3 -> out dir
if [[ $# -gt 0 ]]; then
  SYNTHETIC_IMAGE_SIZE="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  SYNTHETIC_NUM_IMAGES="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  SYNTHETIC_OUT_DIR="$1"
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

# Synthetic out dir arg (optional)
SYNTHETIC_OUT_DIR_ARG=()
if [[ -n "${SYNTHETIC_OUT_DIR}" ]]; then
  SYNTHETIC_OUT_DIR_ARG=(--synthetic-out-dir "${SYNTHETIC_OUT_DIR}")
fi

cd "${ROOT_DIR}"

echo "[run_qwen_vl_synthetic] MODEL=${MODEL}"
echo "[run_qwen_vl_synthetic] MODEL_ID=${MODEL_ID}"
echo "[run_qwen_vl_synthetic] BACKEND=${BACKEND}"
echo "[run_qwen_vl_synthetic] BASE_URL=${BASE_URL}"
echo "[run_qwen_vl_synthetic] API=${API}"
echo "[run_qwen_vl_synthetic] IMAGE_TRANSPORT=${IMAGE_TRANSPORT}"
echo "[run_qwen_vl_synthetic] TP_SIZE=${TP_SIZE}"
echo "[run_qwen_vl_synthetic] DP_SIZE=${DP_SIZE}"
echo "[run_qwen_vl_synthetic] MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "[run_qwen_vl_synthetic] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[run_qwen_vl_synthetic] PROMPT=${PROMPT}"
echo "[run_qwen_vl_synthetic] SYNTHETIC_IMAGE_SIZE=${SYNTHETIC_IMAGE_SIZE}"
echo "[run_qwen_vl_synthetic] SYNTHETIC_NUM_IMAGES=${SYNTHETIC_NUM_IMAGES}"
echo "[run_qwen_vl_synthetic] SYNTHETIC_SEED=${SYNTHETIC_SEED}"
echo "[run_qwen_vl_synthetic] SYNTHETIC_OUT_DIR=${SYNTHETIC_OUT_DIR:-<tmp>}"
echo "[run_qwen_vl_synthetic] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_qwen_vl_synthetic] DEVICE=${DEVICE}"
echo "[run_qwen_vl_synthetic] DTYPE=${DTYPE}"
echo "[run_qwen_vl_synthetic] USE_AMX=${USE_AMX}"
echo "[run_qwen_vl_synthetic] PRINT_MODEL_INFO=${PRINT_MODEL_INFO}"
echo "[run_qwen_vl_synthetic] WARMUP=${WARMUP}"
echo "[run_qwen_vl_synthetic] PROFILE=${PROFILE}"
echo "[run_qwen_vl_synthetic] PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES}"
echo "[run_qwen_vl_synthetic] PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES}"
echo "[run_qwen_vl_synthetic] PROFILE_OUT_DIR=${PROFILE_OUT_DIR}"
echo "[run_qwen_vl_synthetic] PROFILE_OUT_NAME=${PROFILE_OUT_NAME}"
echo "[run_qwen_vl_synthetic] PROFILE_STRICT=${PROFILE_STRICT}"

if [[ ${#PROFILE_ARGS[@]} -gt 0 ]]; then
  printf '[run_qwen_vl_synthetic] PROFILE_ARGS='; printf '%q ' "${PROFILE_ARGS[@]}"; printf '\n'
else
  echo "[run_qwen_vl_synthetic] PROFILE_ARGS=<none>"
fi

if [[ $# -gt 0 ]]; then
  printf '[run_qwen_vl_synthetic] EXTRA_ARGS='; printf '%q ' "$@"; printf '\n'
else
  echo "[run_qwen_vl_synthetic] EXTRA_ARGS=<none>"
fi

python scripts/py/run_vl.py \
  --model "${MODEL}" \
  --model-id "${MODEL_ID}" \
  --backend "${BACKEND}" \
  --dataset synthetic \
  --synthetic-image-size "${SYNTHETIC_IMAGE_SIZE}" \
  --synthetic-num-images "${SYNTHETIC_NUM_IMAGES}" \
  --synthetic-seed "${SYNTHETIC_SEED}" \
  "${SYNTHETIC_OUT_DIR_ARG[@]}" \
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
