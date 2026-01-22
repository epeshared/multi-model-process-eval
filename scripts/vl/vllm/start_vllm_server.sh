#!/usr/bin/env bash
set -euo pipefail

# vLLM OpenAI-compatible server for Qwen2.5-VL (vision-language)
# Usage:
#   MODEL_DIR=/path/to/Qwen2.5-VL-7B-Instruct ./start_vllm_server.sh
# Then point the runner at it:
#   BACKEND=vllm-http BASE_URL=http://127.0.0.1:30000 ./../run_qwen_vl_synthetic.sh
#
# NOTE: The client must send a `model` value that matches SERVED_MODEL_NAME.
# In this repo, backend=vllm-http uses MODEL_ID as the `model` field. Make sure:
#   MODEL_ID == SERVED_MODEL_NAME

# ===== WORK_HOME (scripts/vl) =====
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-VL-7B-Instruct"}
###############################################
echo "Using model: $MODEL_DIR"

# IMPORTANT: must match what client sends as `model` for OpenAI-compatible endpoints.
# Default to MODEL_DIR to match run_qwen_vl_synthetic.sh's default MODEL_ID.
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$MODEL_DIR"}

# ===== Server bind =====
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}

# ===== Parallel / memory =====
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
DTYPE=${DTYPE:-auto}  # auto|float16|bfloat16

# ===== CUDA Graph (optional) =====
# Set CUDA_GRAPH=1 to enable if supported by your vLLM build.
CUDA_GRAPH=${CUDA_GRAPH:-0}

# ===== Multimodal limits (optional) =====
# Example: LIMIT_MM_PER_PROMPT_IMAGE=8
LIMIT_MM_PER_PROMPT_IMAGE=${LIMIT_MM_PER_PROMPT_IMAGE:-""}

# ===== Logging =====
LOG_DIR=${LOG_DIR:-"$PWD/vllm_logs"}
mkdir -p "$LOG_DIR"

# ===== Quick dependency check =====
echo "python=$(command -v python)"

python -c "import vllm" >/dev/null 2>&1 || {
  echo "ERROR: vLLM is not installed in this environment."
  echo "Install with: pip install vllm (or your pinned requirements)."
  exit 1
}

# Print key versions for easier debugging.
python - <<'PY'
import importlib.metadata as m

def v(name: str) -> str:
  try:
    return m.version(name)
  except Exception:
    return "<not installed>"

print("vllm", v("vllm"))
print("openai", v("openai"))
print("torch", v("torch"))
PY

# Preflight import check.
python - <<'PY'
import sys

try:
  from openai.types.chat import ChatCompletionFunctionToolParam  # noqa: F401
except Exception as e:
  print("ERROR: openai package is incompatible with vLLM OpenAI server.")
  print(f"  Import failed: {e}")
  print("  Fix: pip install -U openai")
  sys.exit(1)

try:
  import vllm.entrypoints.openai.api_server  # noqa: F401
except Exception as e:
  print("ERROR: vLLM OpenAI api_server failed to import.")
  print(f"  Import failed: {e}")
  print("  If this mentions openai.types.*, upgrade openai: pip install -U openai")
  sys.exit(1)
PY

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Detect whether vLLM api_server supports --enable-cuda-graph
VLLM_HELP_TEXT="$(python -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"
CUDA_GRAPH_ARG=()
case "${CUDA_GRAPH}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ -n "${VLLM_HELP_TEXT}" ]] && grep -Eq -- '^\s*--enable-cuda-graph\b' <<<"${VLLM_HELP_TEXT}"; then
      CUDA_GRAPH_ARG=(--enable-cuda-graph)
    else
      echo "WARN: vLLM api_server has no --enable-cuda-graph flag; skipping CUDA graph." >&2
    fi
    ;;
esac

CMD=(
python -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --enable-cuda-graph
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype float16
)

if [[ -n "$LIMIT_MM_PER_PROMPT_IMAGE" ]]; then
  CMD+=(--limit-mm-per-prompt "image=$LIMIT_MM_PER_PROMPT_IMAGE")
fi

echo "Launching vLLM server:"
printf '  %q' "${CMD[@]}"
echo

echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Logs: $LOG_DIR/vllm_server.log"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/vllm_server.log"
