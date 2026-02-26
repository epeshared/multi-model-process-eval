#!/usr/bin/env bash
set -euo pipefail

# Run embeddings on a synthetic dataset with controlled input size.
#
# Modes:
#   MODE=input_len (default)  -> dataset=synthetic_fixed_len, exact SYNTHETIC_INPUT_LEN characters
#   MODE=token_len            -> dataset=synthetic_tokens, ~SYNTHETIC_TOKEN_LEN tokens
#
# Environment overrides:
#   MODEL (default: qwen3-embedding-4b)
#   MODEL_ID (default: /home/xtang/models/Qwen/Qwen3-Embedding-4B)
#   BACKEND (default: sglang-offline)
#   MODE (default: input_len)
#   SYNTHETIC_INPUT_LEN (for MODE=input_len; e.g. 512)
#   SYNTHETIC_TOKEN_LEN (for MODE=token_len; e.g. 20)
#   SYNTHETIC_SEED (default: 12345)
#   MAX_SAMPLES (default: 1000)
#   BATCH_SIZE (default: 100)
#   DEVICE (default: cpu)
#   DTYPE (default: bfloat16)
#   USE_AMX (TRUE/1/yes/on to enable)
#   BASE_URL (default: http://127.0.0.1:30000)
#   EMBEDDING_HTTP_TIMEOUT (default: 120)      # passed to run_embedding.py --timeout
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
#   # Fixed character length
#   MODE=input_len SYNTHETIC_INPUT_LEN=512 MAX_SAMPLES=10000 $0 [extra python args...]
#   $0 512 [extra python args...]
#
#   # Fixed token length
#   MODE=token_len SYNTHETIC_TOKEN_LEN=20 MAX_SAMPLES=10000 $0 [extra python args...]
#   $0 20 [extra python args...]

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# MODEL=${MODEL:-qwen3-embedding-4b}
# MODEL_ID=${MODEL_ID:-/home/xtang/models/Qwen/Qwen3-Embedding-4B}

MODEL=${MODEL:-youtu-embedding}
BACKEND=${BACKEND:-sglang}

# Default MODEL_ID differs by backend for this model:
# - offline backends expect a local model path
# - vllm-http expects the served model name (the `model` string used in /v1/embeddings)
if [[ "${MODEL}" == "youtu-embedding" ]]; then
  YOUTU_MODEL_DIR_DEFAULT="/mnt/nvme2n1p1/xtang/models/tencent/youtu-embedding-fp16"
  YOUTU_VLLM_SERVED_MODEL_NAME_DEFAULT="sn-large-multi-language-v0.2.5"
  MODEL_ID="${MODEL_ID:-}"
  if [[ -z "${MODEL_ID}" ]]; then
    case "${BACKEND}" in
      vllm-http|vllm_openai|vllm-http-openai)
        MODEL_ID="${YOUTU_VLLM_SERVED_MODEL_NAME_DEFAULT}"
        ;;
      *)
        MODEL_ID="${YOUTU_MODEL_DIR_DEFAULT}"
        ;;
    esac
  fi
else
  MODEL_ID="${MODEL_ID:-}"
fi

# If the caller provides MODEL_DIR, prefer it for offline backends.
# This avoids accidentally treating a served model name (e.g. "Qwen3-Embedding-4B")
# as a HuggingFace repo_id in offline mode.
if [[ -n "${MODEL_DIR:-}" ]]; then
  case "${BACKEND}" in
    sglang-offline|sglang_offline|vllm|vllm-offline|vllm_offline)
      if [[ -z "${MODEL_ID:-}" ]] || [[ ! -d "${MODEL_ID}" ]]; then
        MODEL_ID="${MODEL_DIR}"
      fi
      ;;
  esac
fi

MAX_SAMPLES=${MAX_SAMPLES:-1000}
BATCH_SIZE=${BATCH_SIZE:-100}
DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-bfloat16}
USE_AMX=${USE_AMX:-TRUE}
BASE_URL=${BASE_URL:-http://127.0.0.1:9090}
EMBEDDING_HTTP_TIMEOUT=${EMBEDDING_HTTP_TIMEOUT:-120}
WARMUP_SAMPLES=${WARMUP_SAMPLES:-1}
PRINT_MODEL_INFO=${PRINT_MODEL_INFO:-0}

# vLLM OpenAI embeddings support encoding_format=float|base64.
# base64 avoids server-side JSON serialization failures when embeddings contain NaNs.
ENCODING_FORMAT=${ENCODING_FORMAT:-}
if [[ "${BACKEND}" == vllm-http* ]] && [[ -z "${ENCODING_FORMAT}" ]]; then
  ENCODING_FORMAT=base64
fi

MODE=${MODE:-input_len}

SYNTHETIC_INPUT_LEN=${SYNTHETIC_INPUT_LEN:-512}
SYNTHETIC_TOKEN_LEN=${SYNTHETIC_TOKEN_LEN:-20}
SYNTHETIC_SEED=${SYNTHETIC_SEED:-12345}

case "${MODE}" in
  token_len|tokens|token|tok)
    MODE=token_len
    ;;
  input_len|input|len|char|chars)
    MODE=input_len
    ;;
  *)
    echo "Error: MODE must be input_len or token_len (got: ${MODE})." >&2
    exit 1
    ;;
esac

# Optional positional override: first arg = length (meaning depends on MODE)
if [[ $# -gt 0 ]] && [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  if [[ "${MODE}" == "token_len" ]]; then
    SYNTHETIC_TOKEN_LEN="$1"
  else
    SYNTHETIC_INPUT_LEN="$1"
  fi
  shift
fi

if [[ "${MODE}" == "token_len" ]]; then
  if [[ ! "${SYNTHETIC_TOKEN_LEN}" =~ ^[0-9]+$ ]] || (( SYNTHETIC_TOKEN_LEN <= 0 )); then
    echo "Usage: MODE=token_len SYNTHETIC_TOKEN_LEN=20 MAX_SAMPLES=10000 $0 [extra python args...]" >&2
    echo "   or: MODE=token_len $0 20 [extra python args...]" >&2
    echo "Error: SYNTHETIC_TOKEN_LEN must be > 0 (got: ${SYNTHETIC_TOKEN_LEN})." >&2
    exit 1
  fi
else
  if [[ ! "${SYNTHETIC_INPUT_LEN}" =~ ^[0-9]+$ ]] || (( SYNTHETIC_INPUT_LEN <= 0 )); then
    echo "Usage: MODE=input_len SYNTHETIC_INPUT_LEN=512 MAX_SAMPLES=10000 $0 [extra python args...]" >&2
    echo "   or: MODE=input_len $0 512 [extra python args...]" >&2
    echo "Error: SYNTHETIC_INPUT_LEN must be > 0 (got: ${SYNTHETIC_INPUT_LEN})." >&2
    exit 1
  fi
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

# HTTP timeout for server-backed backends (sglang/vllm-http).
TIMEOUT_ARG=()
if [[ -n "${EMBEDDING_HTTP_TIMEOUT:-}" ]]; then
  if [[ " $* " != *" --timeout "* ]]; then
    TIMEOUT_ARG=(--timeout "${EMBEDDING_HTTP_TIMEOUT}")
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
echo "[run_embedding_synth] MODE=${MODE}"
if [[ "${MODE}" == "token_len" ]]; then
  echo "[run_embedding_synth] SYNTHETIC_TOKEN_LEN=${SYNTHETIC_TOKEN_LEN}"
else
  echo "[run_embedding_synth] SYNTHETIC_INPUT_LEN=${SYNTHETIC_INPUT_LEN}"
fi
echo "[run_embedding_synth] SYNTHETIC_SEED=${SYNTHETIC_SEED}"
echo "[run_embedding_synth] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_embedding_synth] DEVICE=${DEVICE:-<unset>}"
echo "[run_embedding_synth] DTYPE=${DTYPE:-<unset>}"
echo "[run_embedding_synth] USE_AMX=${USE_AMX}"
echo "[run_embedding_synth] BASE_URL=${BASE_URL:-<unset>}"
echo "[run_embedding_synth] EMBEDDING_HTTP_TIMEOUT=${EMBEDDING_HTTP_TIMEOUT:-<unset>}"
echo "[run_embedding_synth] WARMUP_SAMPLES=${WARMUP_SAMPLES}"
echo "[run_embedding_synth] PRINT_MODEL_INFO=${PRINT_MODEL_INFO}"
echo "[run_embedding_synth] PROFILE=${PROFILE}"
echo "[run_embedding_synth] PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES}"
echo "[run_embedding_synth] PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES}"
echo "[run_embedding_synth] PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-<unset>}"
echo "[run_embedding_synth] PROFILE_OUT_NAME=${PROFILE_OUT_NAME}"
echo "[run_embedding_synth] PROFILE_STRICT=${PROFILE_STRICT}"
echo "[run_embedding_synth] ENCODING_FORMAT=${ENCODING_FORMAT:-<unset>}"

if [[ $# -gt 0 ]]; then
  printf '[run_embedding_synth] EXTRA_ARGS='; printf '%q ' "$@"; printf '\n'
else
  echo "[run_embedding_synth] EXTRA_ARGS=<none>"
fi

DATASET_ARG=()
LEN_ARG=()
if [[ "${MODE}" == "token_len" ]]; then
  DATASET_ARG=(--dataset synthetic_tokens)
  LEN_ARG=(--synthetic-token-len "${SYNTHETIC_TOKEN_LEN}")
else
  DATASET_ARG=(--dataset synthetic_fixed_len)
  LEN_ARG=(--synthetic-input-len "${SYNTHETIC_INPUT_LEN}")
fi

# Choose Python interpreter for the client benchmark.
# Prefer a caller-provided interpreter so dependencies (e.g. torch) are consistent.
EMBEDDING_PYTHON_BIN="${EMBEDDING_PYTHON:-${SGLANG_PYTHON:-python}}"

"${EMBEDDING_PYTHON_BIN}" scripts/embedding/run_embedding.py \
  --model "${MODEL}" \
  "${MODEL_ID_ARG[@]}" \
  --backend "${BACKEND}" \
  "${DATASET_ARG[@]}" \
  --max-samples "${MAX_SAMPLES}" \
  "${LEN_ARG[@]}" \
  --synthetic-seed "${SYNTHETIC_SEED}" \
  --batch-size "${BATCH_SIZE}" \
  "${USE_AMX_ARG[@]}" \
  "${WARMUP_ARG[@]}" \
  "${PRINT_MODEL_INFO_ARG[@]}" \
  "${PROFILE_ARGS[@]}" \
  "${TIMEOUT_ARG[@]}" \
  ${BASE_URL:+--base-url "${BASE_URL}"} \
  ${DEVICE:+--device "${DEVICE}"} \
  ${DTYPE:+--dtype "${DTYPE}"} \
  ${ENCODING_FORMAT:+--encoding-format "${ENCODING_FORMAT}"} \
  "$@"
