#!/usr/bin/env bash
set -euo pipefail

# SGLang server for Qwen3 (LLM)
# Usage:
#   MODEL_DIR=/path/to/Qwen3-0.6B ./start_sglang_server.sh
# Then point client to:
#   BASE_URL=http://127.0.0.1:30000

# ===== WORK_HOME (scripts/qwen3) =====
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen3-0.6B"}
###############################################
echo "Using model: $MODEL_DIR"

# ===== Serve mode =====
DEVICE=${DEVICE:-cpu}   # cpu|cuda
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-${SGLANG_MAX_TOTAL_TOKENS:-65536}}

# ===== CPU engine toggles (optional) =====
if [[ "${DEVICE}" == "cpu" ]]; then
  export SGLANG_USE_CPU_ENGINE=1
fi

# ===== Logging =====
mkdir -p "${WORK_HOME}/sglang/sglang_logs/sglang_cpu"
export SGLANG_TORCH_PROFILER_DIR="${WORK_HOME}/sglang/sglang_logs/sglang_cpu"

echo "Batch size = $BATCH_SIZE"

EXTRA_ARGS=()
if [[ "${DEVICE}" == "cpu" ]]; then
  EXTRA_ARGS+=(--disable-cuda-graph --disable-piecewise-cuda-graph)
fi

# Use the active Python environment.
# - If you want to force a specific interpreter, set SGLANG_PYTHON.
# - Or set SGLANG_CONDA_ENV and we will launch through `conda run -n <env> python`.
PYTHON_CMD=()
if [[ -n "${SGLANG_PYTHON:-}" ]]; then
  PYTHON_CMD=("${SGLANG_PYTHON}")
elif [[ -n "${SGLANG_CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (needed for SGLANG_CONDA_ENV=${SGLANG_CONDA_ENV})" >&2
    exit 127
  fi
  PYTHON_CMD=(conda run -n "${SGLANG_CONDA_ENV}" python)
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

PY_PREFIX="$(${PYTHON_CMD[@]} -c 'import sys; from pathlib import Path; p=Path(sys.executable).resolve(); print(p.parents[1])' 2>/dev/null || true)"
echo "PY_PREFIX=${PY_PREFIX}"

# ===== Preload libraries (best-effort; can be strict with SGLANG_REQUIRE_IOMP=1) =====
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

LIB_TCMALLOC="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
LIB_TBBMALLOC="/usr/lib/x86_64-linux-gnu/libtbbmalloc.so.2"

if [[ ! -f "${LIB_TCMALLOC}" ]]; then
  echo "[start_sglang_server] ERROR: required library not found: ${LIB_TCMALLOC}" >&2
  exit 1
fi
if [[ ! -f "${LIB_TBBMALLOC}" ]]; then
  echo "[start_sglang_server] ERROR: required library not found: ${LIB_TBBMALLOC}" >&2
  exit 1
fi

SGLANG_REQUIRE_IOMP="${SGLANG_REQUIRE_IOMP:-0}"

_find_libiomp() {
  local cand=""

  if [[ -n "${SGLANG_LIB_IOMP:-}" && -f "${SGLANG_LIB_IOMP}" ]]; then
    echo "${SGLANG_LIB_IOMP}"
    return 0
  fi

  for cand in "${PY_PREFIX}/lib/libiomp5.so" "${PY_PREFIX}/lib/libiomp5.so.5"; do
    if [[ -f "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done

  for cand in "/usr/lib/x86_64-linux-gnu/libiomp5.so" "/usr/lib/x86_64-linux-gnu/libiomp5.so.5"; do
    if [[ -f "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done

  if command -v ldconfig >/dev/null 2>&1; then
    cand="$(ldconfig -p 2>/dev/null | awk '/libiomp5\\.so/{print $NF; exit}')" || true
    if [[ -n "${cand}" && -f "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  fi

  local envs_dir conda_root
  envs_dir="$(dirname "${PY_PREFIX}")"
  conda_root="$(dirname "${envs_dir}")"
  if [[ -d "${conda_root}/envs" ]]; then
    shopt -s nullglob
    for cand in "${conda_root}/envs"/*/lib/libiomp5.so "${conda_root}/envs"/*/lib/libiomp5.so.5; do
      if [[ -f "${cand}" ]]; then
        echo "${cand}"
        shopt -u nullglob
        return 0
      fi
    done
    shopt -u nullglob
  fi

  return 1
}

LIB_IOMP=""
if LIB_IOMP="$(_find_libiomp)"; then
  :
else
  LIB_IOMP=""
fi

if [[ -z "${LIB_IOMP}" ]]; then
  if [[ "${SGLANG_REQUIRE_IOMP}" == "1" || "${SGLANG_REQUIRE_IOMP}" == "true" ]]; then
    echo "[start_sglang_server] ERROR: libiomp5.so not found (set SGLANG_LIB_IOMP or install intel-openmp)" >&2
    exit 1
  fi
  echo "[start_sglang_server] WARN: libiomp5.so not found; continuing without it" >&2
  export LD_PRELOAD="${LIB_TCMALLOC}:${LIB_TBBMALLOC}${LD_PRELOAD:+:${LD_PRELOAD}}"
else
  echo "[start_sglang_server] Using libiomp: ${LIB_IOMP}" >&2
  export LD_PRELOAD="${LIB_TCMALLOC}:${LIB_TBBMALLOC}:${LIB_IOMP}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

_preload_err="$(LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${LD_PRELOAD}" /usr/bin/true 2>&1)" || {
  echo "[start_sglang_server] ERROR: LD_PRELOAD test failed" >&2
  echo "${_preload_err}" >&2
  exit 1
}
if [[ -n "${_preload_err}" ]]; then
  echo "[start_sglang_server] ERROR: LD_PRELOAD produced loader output" >&2
  echo "${_preload_err}" >&2
  exit 1
fi

"${PYTHON_CMD[@]}" -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --tokenizer-path "$MODEL_DIR" \
  --trust-remote-code \
  --disable-overlap-schedule \
  --device "${DEVICE}" \
  --host "$HOST" --port "$PORT" \
  --skip-server-warmup \
  "${EXTRA_ARGS[@]}" \
  --tp "$TP" \
  --torch-compile-max-bs "$BATCH_SIZE" \
  --enable-tokenizer-batch-encode \
  --log-level error \
  --max-total-tokens "$MAX_TOTAL_TOKENS"
