#!/usr/bin/env bash
set -euo pipefail

# CPU-only vLLM OpenAI-compatible server for embedding models.
#
# This script forces CPU execution:
#   - Always clears CUDA_VISIBLE_DEVICES
#   - Always passes --device cpu if api_server supports it
#   - Never sets --gpu-memory-utilization
#   - Does NOT accept/parse --device CLI
#
# Example:
#   MODEL_DIR=/mnt/nvme2n1p1/xtang/models/tencent/youtu-embedding \
#   SERVED_MODEL_NAME=sn-large-multi-language-v0.2.5 \
#   PORT=9090 \
#   ./start_vllm_server_cpu.sh
#
# Test:
#   curl -X POST --location "http://127.0.0.1:9090/v1/embeddings" \
#     -H "Content-Type: application/json" \
#     -d '{
#           "encoding_format": "float",
#           "input": ["衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买"],
#           "model": "sn-large-multi-language-v0.2.5"
#         }'

EXTRA_VLLM_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage:
  ./start_vllm_server_cpu.sh [extra vLLM args...]

Env overrides:
  MODEL_DIR, SERVED_MODEL_NAME, HOST, PORT, TP, MAX_MODEL_LEN, DTYPE,
  LOG_DIR, QUANTIZATION, DRY_RUN

Optional:
Notes:
  - This script ALWAYS runs on CPU (forces CUDA_VISIBLE_DEVICES="").
  - If your vLLM api_server supports --device, we pass: --device cpu.
EOF
      exit 0
      ;;
    --device|--device=*)
      echo "ERROR: This CPU-only script does not accept --device. It always runs on CPU." >&2
      exit 2
      ;;
    --)
      # Detect optional flags (vLLM versions differ in how embeddings are enabled).
      VLLM_HELP_TEXT="$("${PYTHON_CMD[@]}" -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"

      CONVERT_ARG=()
      if [[ -n "$VLLM_HELP_TEXT" ]] && grep -Eq -- '^\s*--convert\b' <<<"$VLLM_HELP_TEXT"; then
        CONVERT_ARG=(--convert embed)
      fi

      PROMPT_EMBEDS_ARG=()
      if [[ -n "$VLLM_HELP_TEXT" ]] && grep -Eq -- '^\s*--enable-prompt-embeds\b' <<<"$VLLM_HELP_TEXT"; then
        PROMPT_EMBEDS_ARG=(--enable-prompt-embeds)
      fi

      MM_EMBEDS_ARG=()
      if [[ -n "$VLLM_HELP_TEXT" ]] && grep -Eq -- '^\s*--enable-mm-embeds\b' <<<"$VLLM_HELP_TEXT"; then
        MM_EMBEDS_ARG=(--enable-mm-embeds)
      fi

      shift
        "${PYTHON_CMD[@]}" -m vllm.entrypoints.openai.api_server
      break
      ;;
    *)
      EXTRA_VLLM_ARGS+=("$1")
      shift
      ;;
  esac
          "${CONVERT_ARG[@]}"
          "${PROMPT_EMBEDS_ARG[@]}"
          "${MM_EMBEDS_ARG[@]}"
done

# ===== WORK_HOME (scripts/embedding) =====
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
# MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/tencent/youtu-embedding-fp16"}
# MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen3-Embedding-0.6B"}
MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen3-Embedding-4B"}
###############################################
echo "Using model: $MODEL_DIR"

# Make MODEL_DIR visible to Python preflight snippets.
export MODEL_DIR

# IMPORTANT: must match what client sends as `model` for OpenAI-compatible endpoints.
# SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-sn-large-multi-language-v0.2.5}
# SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3-Embedding-0.6B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3-Embedding-4B}

# ===== Server bind =====
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-9090}

# ===== Parallel / memory =====
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

# dtype for CPU. For most Intel CPU setups, bfloat16 is a good default if supported.
# Options depend on your vLLM build: auto|float32|float16|bfloat16
DTYPE=${DTYPE:-bfloat16}

case "${DTYPE}" in
  float16|fp16|half)
    echo "WARN: DTYPE=${DTYPE} on CPU can produce NaNs for embeddings; forcing DTYPE=bfloat16." >&2
    DTYPE=bfloat16
    ;;
esac

# CPU startup can spend a long time compiling/warming up. Default to eager mode
# (no compilation) for predictable startup; set ENFORCE_EAGER=0 to re-enable.
ENFORCE_EAGER=${ENFORCE_EAGER:-1}

# Optional quantization override (only if your vLLM build supports it and model allows).
QUANTIZATION=${QUANTIZATION:-}

# If set, print the final CMD and exit before launching the server.
DRY_RUN=${DRY_RUN:-0}

# ===== Logging =====
LOG_DIR=${LOG_DIR:-"$PWD/vllm_logs"}
mkdir -p "$LOG_DIR"

# ===== Force CPU-only =====
export CUDA_VISIBLE_DEVICES=""
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Allow choosing python via env (portable in scale-test).
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
  if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
  fi
  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: python not found on PATH" >&2
    exit 127
  fi
  PYTHON_CMD=("${PYTHON_BIN}")
fi

echo "PYTHON_BIN=${PYTHON_CMD[*]}"

echo "python=${PYTHON_CMD[*]}"

"${PYTHON_CMD[@]}" -c "import vllm" >/dev/null 2>&1 || {
  echo "ERROR: vLLM is not installed in this environment."
  echo "Install with: pip install vllm (or your pinned requirements)."
  exit 1
}

# Print key versions for easier debugging.
"${PYTHON_CMD[@]}" - <<'PY'
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
"${PYTHON_CMD[@]}" - <<'PY'
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
  sys.exit(1)
PY

# Warn if torch still sees CUDA (you may be in a GPU torch build).
"${PYTHON_CMD[@]}" - <<'PY'
import torch
if torch.cuda.is_available():
    print("WARN: torch.cuda.is_available() == True, but this script forces CPU via CUDA_VISIBLE_DEVICES=''.")
    print("      If you intended CPU-only wheels, install CPU-only torch/vLLM to avoid accidental GPU codepaths.")
else:
    print("OK: torch.cuda.is_available() == False (CPU-only torch).")
PY

# Fail fast for FP8 (float8) checkpoints on CPU.
# Some models don't declare FP8 in config.json, but their weights are stored as float8.
"${PYTHON_CMD[@]}" - <<'PY'
import os
import sys

import torch

model_dir = os.environ.get("MODEL_DIR", "")
path = os.path.join(model_dir, "model.safetensors")
if not model_dir or not os.path.exists(path):
  sys.exit(0)

if torch.cuda.is_available():
  # This script still forces CPU, but skip this specific FP8 check if CUDA is available.
  sys.exit(0)

try:
  from safetensors import safe_open
except Exception:
  sys.exit(0)

probe_keys = [
  "model.layers.0.mlp.down_proj.weight",
  "model.layers.0.self_attn.q_proj.weight",
  "model.layers.0.attention.q_proj.weight",
]

with safe_open(path, framework="pt", device="cpu") as f:
  keys = set(f.keys())
  found = None
  for k in probe_keys:
    if k in keys:
      t = f.get_tensor(k)
      found = (k, str(t.dtype))
      break

if found is not None:
  k, dtype = found
  if "float8" in dtype:
    print("ERROR: This checkpoint stores weights in FP8 (float8), e.g.:")
    print(f"       {k} has dtype={dtype}")
    print("       vLLM CPU builds cannot run FP8 weights (FP8 paths call torch.cuda / GPU kernels).")
    print("")
    print("Fix options:")
    print("  1) Use a CUDA-enabled environment (GPU torch + GPU vLLM) and run on GPU.")
    print("  2) Use a non-FP8 checkpoint (BF16/FP16/FP32) or re-export weights without FP8.")
    sys.exit(1)
PY

# Detect whether vLLM api_server supports --device
VLLM_HELP_TEXT="$("${PYTHON_CMD[@]}" -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"
DEVICE_ARG=()
if [[ -n "${VLLM_HELP_TEXT}" ]] && grep -Eq -- '^\s*--device\b' <<<"${VLLM_HELP_TEXT}"; then
  DEVICE_ARG=(--device cpu)
else
  echo "WARN: vLLM api_server has no --device flag; relying on CUDA_VISIBLE_DEVICES='' only." >&2
fi

# vLLM version compatibility:
# - Older versions exposed /v1/embeddings by default.
# - Newer versions require embedding-mode args (runner=pooling, convert=embed).
RUNNER_ARG=()
if [[ -n "${VLLM_HELP_TEXT}" ]] && grep -Eq -- '^\s*--runner\b' <<<"${VLLM_HELP_TEXT}"; then
  RUNNER_ARG=(--runner pooling)
fi

CONVERT_ARG=()
if [[ -n "${VLLM_HELP_TEXT}" ]] && grep -Eq -- '^\s*--convert\b' <<<"${VLLM_HELP_TEXT}"; then
  CONVERT_ARG=(--convert embed)
fi

PROMPT_EMBEDS_ARG=()
if [[ -n "${VLLM_HELP_TEXT}" ]] && grep -Eq -- '^\s*--enable-prompt-embeds\b' <<<"${VLLM_HELP_TEXT}"; then
  PROMPT_EMBEDS_ARG=(--enable-prompt-embeds)
fi

MM_EMBEDS_ARG=()
if [[ -n "${VLLM_HELP_TEXT}" ]] && grep -Eq -- '^\s*--enable-mm-embeds\b' <<<"${VLLM_HELP_TEXT}"; then
  MM_EMBEDS_ARG=(--enable-mm-embeds)
fi

QUANT_ARG=()
if [[ -n "${QUANTIZATION}" ]]; then
  if [[ " ${EXTRA_VLLM_ARGS[*]} " != *" --quantization "* ]] && [[ " ${EXTRA_VLLM_ARGS[*]} " != *" --quantization="* ]]; then
    QUANT_ARG=(--quantization "${QUANTIZATION}")
  fi
fi

EAGER_ARG=()
case "${ENFORCE_EAGER}" in
  1|true|TRUE|yes|YES|on|ON)
    # Avoid slow compilation warmup on CPU.
    if [[ " ${EXTRA_VLLM_ARGS[*]} " != *" --enforce-eager "* ]] && [[ " ${EXTRA_VLLM_ARGS[*]} " != *" --enforce-eager"* ]]; then
      EAGER_ARG=(--enforce-eager)
    fi
    ;;
esac

TC_PATH="/root/miniforge3/envs/xtang-embedding-cpu/lib/libtcmalloc_minimal.so.4"
IOMP_PATH="/root/miniforge3/envs/xtang-embedding-cpu/lib/libiomp5.so"

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
  "${PYTHON_CMD[@]}" -m vllm.entrypoints.openai.api_server
    --model "$MODEL_DIR"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    "${DEVICE_ARG[@]}"
    "${QUANT_ARG[@]}"
    "${EAGER_ARG[@]}"
    "${RUNNER_ARG[@]}"
    "${CONVERT_ARG[@]}"
    "${PROMPT_EMBEDS_ARG[@]}"
    "${MM_EMBEDS_ARG[@]}"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --dtype "$DTYPE"
)

if [[ ${#EXTRA_VLLM_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_VLLM_ARGS[@]}")
fi

echo "Launching vLLM server (CPU-only):"
printf '  %q' "${CMD[@]}"
echo

case "${DRY_RUN}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "DRY_RUN=1: not launching server."
    exit 0
    ;;
esac

echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Logs: $LOG_DIR/vllm_server.log"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/vllm_server.log"
