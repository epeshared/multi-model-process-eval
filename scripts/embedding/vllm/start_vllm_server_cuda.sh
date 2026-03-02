#!/usr/bin/env bash
set -euo pipefail

# vLLM OpenAI-compatible server for embedding models (CUDA).
#
# This is intended as a drop-in alternative backend to SGLang for scale-test.
# It launches: python -m vllm.entrypoints.openai.api_server ...
#
# Env overrides (commonly set by scale-test job env):
#   MODEL_DIR, SERVED_MODEL_NAME, HOST, PORT, TP, MAX_MODEL_LEN,
#   GPU_MEMORY_UTILIZATION, DTYPE,
#   VLLM_PYTHON, VLLM_CONDA_ENV,
#   LIMIT_MM_PER_PROMPT_IMAGE (optional; forwarded when supported)

EXTRA_VLLM_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage:
  ./start_vllm_server_cuda.sh [extra vLLM args...]

Env overrides:
  MODEL_DIR, SERVED_MODEL_NAME, HOST, PORT, TP, MAX_MODEL_LEN,
  GPU_MEMORY_UTILIZATION, DTYPE,
  VLLM_PYTHON, VLLM_CONDA_ENV,
  LIMIT_MM_PER_PROMPT_IMAGE
EOF
      exit 0
      ;;
    --)
      shift
      EXTRA_VLLM_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_VLLM_ARGS+=("$1")
      shift
      ;;
  esac
done

WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen3-VL-Embedding-2B"}
echo "Using model: $MODEL_DIR"

# IMPORTANT: must match what clients send as `model` for /v1/embeddings.
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$MODEL_DIR"}

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}

TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
DTYPE=${DTYPE:-auto}  # auto|float16|bfloat16

LIMIT_MM_PER_PROMPT_IMAGE=${LIMIT_MM_PER_PROMPT_IMAGE:-""}

LOG_DIR=${LOG_DIR:-"$PWD/vllm_logs"}
mkdir -p "$LOG_DIR"

# Choose python for the server.
PYTHON_CMD=()
if [[ -n "${VLLM_PYTHON:-}" ]]; then
  PYTHON_CMD=("${VLLM_PYTHON}")
elif [[ -n "${VLLM_CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (needed for VLLM_CONDA_ENV=${VLLM_CONDA_ENV})" >&2
    exit 127
  fi
  PYTHON_CMD=(conda run -n "${VLLM_CONDA_ENV}" python)
else
  PYTHON_BIN="$(command -v python || true)"
  if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
  fi
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: python not found on PATH" >&2
    exit 127
  fi
  PYTHON_CMD=("$PYTHON_BIN")
fi

echo "PYTHON_BIN=${PYTHON_CMD[*]}"

"${PYTHON_CMD[@]}" -c "import vllm" >/dev/null 2>&1 || {
  echo "ERROR: vLLM is not installed in this environment." >&2
  exit 1
}

VLLM_HELP_TEXT="$("${PYTHON_CMD[@]}" -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"

# vLLM version compatibility:
# - Older versions exposed /v1/embeddings by default.
# - Newer versions require embedding-mode args (runner=pooling, convert=embed).
RUNNER_ARG=()
if [[ -n "$VLLM_HELP_TEXT" ]] && grep -Fq -- '--runner' <<<"$VLLM_HELP_TEXT"; then
  RUNNER_ARG=(--runner pooling)
fi

CONVERT_ARG=()
if [[ -n "$VLLM_HELP_TEXT" ]] && grep -Fq -- '--convert' <<<"$VLLM_HELP_TEXT"; then
  CONVERT_ARG=(--convert embed)
fi

PROMPT_EMBEDS_ARG=()
if [[ -n "$VLLM_HELP_TEXT" ]] && grep -Fq -- '--enable-prompt-embeds' <<<"$VLLM_HELP_TEXT"; then
  PROMPT_EMBEDS_ARG=(--enable-prompt-embeds)
fi

MM_EMBEDS_ARG=()
if [[ -n "$VLLM_HELP_TEXT" ]] && grep -Fq -- '--enable-mm-embeds' <<<"$VLLM_HELP_TEXT"; then
  MM_EMBEDS_ARG=(--enable-mm-embeds)
fi

LIMIT_MM_ARG=()
if [[ -n "$LIMIT_MM_PER_PROMPT_IMAGE" ]]; then
  if [[ -n "$VLLM_HELP_TEXT" ]] && grep -Fq -- '--limit-mm-per-prompt' <<<"$VLLM_HELP_TEXT"; then
    LIMIT_MM_ARG=(--limit-mm-per-prompt "image=$LIMIT_MM_PER_PROMPT_IMAGE")
  fi
fi

CMD=(
  "${PYTHON_CMD[@]}" -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype "$DTYPE"
  "${RUNNER_ARG[@]}"
  "${CONVERT_ARG[@]}"
  "${PROMPT_EMBEDS_ARG[@]}"
  "${MM_EMBEDS_ARG[@]}"
  "${LIMIT_MM_ARG[@]}"
  "${EXTRA_VLLM_ARGS[@]}"
)

echo "Launching vLLM server (CUDA):"
printf '  %q' "${CMD[@]}"
echo

echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Logs: $LOG_DIR/vllm_server.log"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/vllm_server.log"
