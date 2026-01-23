#!/usr/bin/env bash
set -euo pipefail

# vLLM OpenAI-compatible server for Qwen3 (LLM)
# Usage:
#   MODEL_DIR=/path/to/Qwen3-0.6B ./start_vllm_server.sh
# Then point client at it:
#   BACKEND=vllm-http BASE_URL=http://127.0.0.1:8000 ...

# ===== WORK_HOME (scripts/qwen3) =====
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen3-0.6B"}
###############################################
echo "Using model: $MODEL_DIR"

# IMPORTANT: must match what client sends as `model` for OpenAI-compatible endpoints.
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-0.6b}

# ===== Server bind =====
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# ===== Parallel / memory =====
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
DTYPE=${DTYPE:-auto}  # auto|float16|bfloat16

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

# Optional preloads (tcmalloc + iomp) for better CPU threading/memory behavior.
# Safe under `set -u` and only preloads libs that exist.
PY_PREFIX="$(python - <<'PY'
import sys
print(sys.prefix)
PY
)"

TC_PATH=${TC_PATH:-"${PY_PREFIX}/lib/libtcmalloc_minimal.so.4"}
IOMP_PATH=${IOMP_PATH:-"${PY_PREFIX}/lib/libiomp5.so"}

PRELOAD_ITEMS=()
if [[ -f "${TC_PATH}" ]]; then
  PRELOAD_ITEMS+=("${TC_PATH}")
else
  echo "WARN: TC_PATH not found, skipping preload: ${TC_PATH}" >&2
fi

if [[ -f "${IOMP_PATH}" ]]; then
  PRELOAD_ITEMS+=("${IOMP_PATH}")
else
  echo "WARN: IOMP_PATH not found, skipping preload: ${IOMP_PATH}" >&2
fi

if [[ -n "${LD_PRELOAD:-}" ]]; then
  PRELOAD_ITEMS+=("${LD_PRELOAD}")
fi

if [[ ${#PRELOAD_ITEMS[@]} -gt 0 ]]; then
  export LD_PRELOAD
  LD_PRELOAD="$(IFS=:; echo "${PRELOAD_ITEMS[*]}")"
  echo "LD_PRELOAD=${LD_PRELOAD}"
else
  echo "LD_PRELOAD=<unset>"
fi

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

echo "Launching vLLM server:"
printf '  %q' "${CMD[@]}"
echo

echo "Logs: $LOG_DIR/vllm_server.log"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/vllm_server.log"
