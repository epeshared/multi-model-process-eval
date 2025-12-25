#!/usr/bin/env bash
set -euo pipefail

# Run embeddings on synthetic_tokens dataset only.
# It generates MAX_SAMPLES texts with ~SYNTHETIC_TOKEN_LEN tokens each.
#
# Environment overrides:
#   MODEL (default: qwen3-embedding-4b)
#   MODEL_ID (default: /home/xtang/models/Qwen/Qwen3-Embedding-4B)
#   BACKEND (default: sglang-offline)
#   SYNTHETIC_TOKEN_LEN (required; e.g. 20)
#   SYNTHETIC_SEED (default: 12345)
#   MAX_SAMPLES (default: 1000)
#   BATCH_SIZE (default: 100)
#   DEVICE (default: cpu)
#   DTYPE (default: bfloat16)
#   USE_AMX (TRUE/1/yes/on to enable)
#   BASE_URL (default: http://127.0.0.1:30000)
#   WARMUP_SAMPLES (default: 1)
#   PRINT_MODEL_INFO (true/1/yes/on to enable)
#
# Profiling envs:
#   PROFILE (true/1/yes/on)                    # enables --profile
#   PROFILE_RECORD_SHAPES (true/1/yes/on)      # enables --profile-record-shapes
#   PROFILE_ACTIVITIES (default: CPU,CUDA)     # e.g. CPU,CUDA
#   PROFILE_OUT_DIR                            # --profile-out-dir
#   PROFILE_OUT_NAME (default: embedding_profile)
#   PROFILE_STRICT (true/1/yes/on)             # --profile-strict
#
# Usage:
#   SYNTHETIC_TOKEN_LEN=20 MAX_SAMPLES=10000 ./run_embedding_synthetic.sh [extra python args...]
#   ./run_embedding_synthetic.sh 20 [extra python args...]

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL=${MODEL:-qwen3-embedding-4b}
MODEL_ID=${MODEL_ID:-/home/xtang/models/Qwen/Qwen3-Embedding-4B}
BACKEND=${BACKEND:-sglang-offline}

MAX_SAMPLES=${MAX_SAMPLES:-1000}
BATCH_SIZE=${BATCH_SIZE:-100}
DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-bfloat16}
USE_AMX=${USE_AMX:-TRUE}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
WARMUP_SAMPLES=${WARMUP_SAMPLES:-1}
PRINT_MODEL_INFO=${PRINT_MODEL_INFO:-0}

SYNTHETIC_TOKEN_LEN=${SYNTHETIC_TOKEN_LEN:-0}
SYNTHETIC_SEED=${SYNTHETIC_SEED:-12345}

# Optional positional override: first arg = token_len
if [[ $# -gt 0 ]] && [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  SYNTHETIC_TOKEN_LEN="$1"
  shift
fi

if [[ ! "${SYNTHETIC_TOKEN_LEN}" =~ ^[0-9]+$ ]] || (( SYNTHETIC_TOKEN_LEN <= 0 )); then
  echo "Usage: SYNTHETIC_TOKEN_LEN=20 MAX_SAMPLES=10000 $0 [extra python args...]" >&2
  echo "   or: $0 20 [extra python args...]" >&2
  echo "Error: SYNTHETIC_TOKEN_LEN must be > 0 (got: ${SYNTHETIC_TOKEN_LEN})." >&2
  exit 1
fi

if [[ ! "${MAX_SAMPLES}" =~ ^-?[0-9]+$ ]] || (( MAX_SAMPLES <= 0 )); then
  echo "Error: MAX_SAMPLES must be > 0 (got: ${MAX_SAMPLES})." >&2
  exit 1
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
# Profiling args
# - If user already passed the flag in EXTRA_ARGS, don't add duplicates.
# -------------------------
PROFILE=${PROFILE:-0}
PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-0}
PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES:-CPU,CUDA}
PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-}
PROFILE_OUT_NAME=${PROFILE_OUT_NAME:-embedding_profile}
PROFILE_STRICT=${PROFILE_STRICT:-0}

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

echo "[run_embedding_synth] MODEL=${MODEL}"
echo "[run_embedding_synth] MODEL_ID=${MODEL_ID:-<unset>}"
echo "[run_embedding_synth] BACKEND=${BACKEND}"
echo "[run_embedding_synth] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[run_embedding_synth] SYNTHETIC_TOKEN_LEN=${SYNTHETIC_TOKEN_LEN}"
echo "[run_embedding_synth] SYNTHETIC_SEED=${SYNTHETIC_SEED}"
echo "[run_embedding_synth] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_embedding_synth] DEVICE=${DEVICE:-<unset>}"
echo "[run_embedding_synth] DTYPE=${DTYPE:-<unset>}"
echo "[run_embedding_synth] USE_AMX=${USE_AMX}"
echo "[run_embedding_synth] BASE_URL=${BASE_URL:-<unset>}"
echo "[run_embedding_synth] WARMUP_SAMPLES=${WARMUP_SAMPLES}"
echo "[run_embedding_synth] PRINT_MODEL_INFO=${PRINT_MODEL_INFO}"
echo "[run_embedding_synth] PROFILE=${PROFILE}"
echo "[run_embedding_synth] PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES}"
echo "[run_embedding_synth] PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES}"
echo "[run_embedding_synth] PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-<unset>}"
echo "[run_embedding_synth] PROFILE_OUT_NAME=${PROFILE_OUT_NAME}"
echo "[run_embedding_synth] PROFILE_STRICT=${PROFILE_STRICT}"

if [[ $# -gt 0 ]]; then
  printf '[run_embedding_synth] EXTRA_ARGS='; printf '%q ' "$@"; printf '\n'
else
  echo "[run_embedding_synth] EXTRA_ARGS=<none>"
fi

python scripts/py/run_embedding.py \
  --model "${MODEL}" \
  "${MODEL_ID_ARG[@]}" \
  --backend "${BACKEND}" \
  --dataset synthetic_tokens \
  --max-samples "${MAX_SAMPLES}" \
  --synthetic-token-len "${SYNTHETIC_TOKEN_LEN}" \
  --synthetic-seed "${SYNTHETIC_SEED}" \
  --batch-size "${BATCH_SIZE}" \
  "${USE_AMX_ARG[@]}" \
  "${WARMUP_ARG[@]}" \
  "${PRINT_MODEL_INFO_ARG[@]}" \
  "${PROFILE_ARGS[@]}" \
  ${BASE_URL:+--base-url "${BASE_URL}"} \
  ${DEVICE:+--device "${DEVICE}"} \
  ${DTYPE:+--dtype "${DTYPE}"} \
  "$@"
