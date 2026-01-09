#!/usr/bin/env bash
set -euo pipefail

# vLLM OpenAI-compatible server for Qwen2.5-Omni
# Usage:
#   MODEL_DIR=/path/to/Qwen2.5-Omni-7B ./start_vllm_server.sh
# Then point the omni runner at it:
#   BACKEND=vllm-http BASE_URL=http://127.0.0.1:8000 ./../run_qwen_omni_synthetic.sh 64x64 1

# ===== WORK_HOME (repo/scripts root) =====
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-Omni-3B"}
###############################################
echo "Using model: $MODEL_DIR"

# IMPORTANT: this name must match what the client sends as `model` for OpenAI-compatible endpoints.
# In this repo, for backend=vllm-http we send the CLI `--model` value (e.g. qwen2.5-omni-3b).
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen2.5-omni-3b}

# ===== Server bind =====
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# ===== Parallel / memory =====
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
DTYPE=${DTYPE:-auto}  # auto|float16|bfloat16

# ===== Multimodal (optional) =====
# vLLM multimodal settings vary by version/model. Keep this optional and override if needed.
# Examples (if your vLLM build supports it):
#   LIMIT_MM_PER_PROMPT_IMAGE=8
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
# vLLM's OpenAI server depends on the `openai` Python package for type schemas.
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

CMD=(
python -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype "$DTYPE"
)

if [[ -n "$LIMIT_MM_PER_PROMPT_IMAGE" ]]; then
  CMD+=(--limit-mm-per-prompt "image=$LIMIT_MM_PER_PROMPT_IMAGE")
fi

echo "Launching vLLM server:"
printf '  %q' "${CMD[@]}"
echo

echo "Logs: $LOG_DIR/vllm_server.log"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/vllm_server.log"
