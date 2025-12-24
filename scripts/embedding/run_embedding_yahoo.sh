#!/usr/bin/env bash
set -euo pipefail

# Run embeddings on the yahoo_answers JSONL dataset using scripts/py/run_embedding.py.
#
# Environment overrides:
#   MODEL (default: qwen3-embedding-4b)
#   MODEL_ID (default: empty, uses preset mapping in run_embedding.py)
#   BACKEND (default: sglang-offline)
#   YAHOO_MODE (default: q)  # q | a | q+a
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

MODEL=${MODEL:-qwen3-embedding-4b}
MODEL_ID=${MODEL_ID:-/home/xtang/models/Qwen/Qwen3-Embedding-4B}
DATASET_PATH=${DATASET_PATH:-/home/xtang/datasets/yahoo_answers_title_answer.jsonl}

# Positional override for dataset path (optional if DATASET_PATH is already set)
if [[ $# -gt 0 ]]; then
  DATASET_PATH="$1"
  shift
fi

if [[ -z "${DATASET_PATH}" ]]; then
  echo "Usage: $0 <yahoo_answers.jsonl> [extra python args...]" >&2
  echo "Hint: set DATASET_PATH env var or pass the path as the first argument." >&2
  exit 1
fi

BACKEND=${BACKEND:-sglang-offline}
YAHOO_MODE=${YAHOO_MODE:-q}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
BATCH_SIZE=${BATCH_SIZE:-100}
DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-bfloat16}
USE_AMX=${USE_AMX:-TRUE}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
WARMUP_SAMPLES=${WARMUP_SAMPLES:-1}
PRINT_MODEL_INFO=${PRINT_MODEL_INFO:-0}

# Profiling envs (new)
PROFILE=${PROFILE:-0}
PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-0}
PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES:-CPU,CUDA}
PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-}
PROFILE_OUT_NAME=${PROFILE_OUT_NAME:-embedding_profile}
PROFILE_STRICT=${PROFILE_STRICT:-0}

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

WARMUP_ARG=()
if [[ "${WARMUP_SAMPLES}" =~ ^[0-9]+$ ]] && (( WARMUP_SAMPLES > 1 )); then
  WARMUP_ARG=(--warmup-samples "${WARMUP_SAMPLES}")
fi

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

cd "${ROOT_DIR}"

echo "[run_embedding_yahoo] MODEL=${MODEL}"
echo "[run_embedding_yahoo] MODEL_ID=${MODEL_ID:-<unset>}"
echo "[run_embedding_yahoo] BACKEND=${BACKEND}"
echo "[run_embedding_yahoo] YAHOO_MODE=${YAHOO_MODE}"
echo "[run_embedding_yahoo] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[run_embedding_yahoo] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_embedding_yahoo] DEVICE=${DEVICE:-<unset>}"
echo "[run_embedding_yahoo] DTYPE=${DTYPE:-<unset>}"
echo "[run_embedding_yahoo] USE_AMX=${USE_AMX}"
echo "[run_embedding_yahoo] BASE_URL=${BASE_URL:-<unset>}"
echo "[run_embedding_yahoo] WARMUP_SAMPLES=${WARMUP_SAMPLES}"
echo "[run_embedding_yahoo] PRINT_MODEL_INFO=${PRINT_MODEL_INFO}"
echo "[run_embedding_yahoo] PROFILE=${PROFILE}"
echo "[run_embedding_yahoo] PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES}"
echo "[run_embedding_yahoo] PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES}"
echo "[run_embedding_yahoo] PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-<unset>}"
echo "[run_embedding_yahoo] PROFILE_OUT_NAME=${PROFILE_OUT_NAME}"
echo "[run_embedding_yahoo] PROFILE_STRICT=${PROFILE_STRICT}"
echo "[run_embedding_yahoo] DATASET_PATH=${DATASET_PATH}"
if [[ $# -gt 0 ]]; then
  printf '[run_embedding_yahoo] EXTRA_ARGS='; printf '%q ' "$@"; printf '\n'
else
  echo "[run_embedding_yahoo] EXTRA_ARGS=<none>"
fi

python scripts/py/run_embedding.py \
  --model "${MODEL}" \
  "${MODEL_ID_ARG[@]}" \
  --backend "${BACKEND}" \
  --dataset yahoo_answers \
  --dataset-path "${DATASET_PATH}" \
  --yahoo-mode "${YAHOO_MODE}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  "${USE_AMX_ARG[@]}" \
  "${WARMUP_ARG[@]}" \
  "${PRINT_MODEL_INFO_ARG[@]}" \
  "${PROFILE_ARGS[@]}" \
  ${BASE_URL:+--base-url "${BASE_URL}"} \
  ${DEVICE:+--device "${DEVICE}"} \
  ${DTYPE:+--dtype "${DTYPE}"} \
  "$@"
