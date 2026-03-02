#!/usr/bin/env bash
set -euo pipefail

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-base-patch32"
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-large-patch14-336"
MODEL_DIR=${MODEL_DIR:-"/home/xtang//models/Qwen/Qwen3-Embedding-4B"}
# MODEL_DIR="$WORK_HOME/models/Qwen/Qwen3-Embedding-0.6B"
###############################################
echo "Using model: $MODEL_DIR"


# Use the active Python environment.
# - If you want to force a specific interpreter, set SGLANG_PYTHON.
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

PY_PREFIX="$("${PYTHON_CMD[@]}" -c 'import sys; from pathlib import Path; p=Path(sys.executable).resolve(); print(p.parents[1])' 2>/dev/null || true)"
echo "PY_PREFIX=${PY_PREFIX}"

# ===== Preload libraries (best-effort; can be strict with SGLANG_REQUIRE_IOMP=1) =====
# Keep system libs available, but do not clobber any existing LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

LIB_TCMALLOC="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
LIB_TBBMALLOC="/usr/lib/x86_64-linux-gnu/libtbbmalloc.so.2"

if [[ ! -f "${LIB_TCMALLOC}" ]]; then
   echo "[start_sglang_server_cuda] ERROR: required library not found: ${LIB_TCMALLOC}" >&2
   exit 1
fi
if [[ ! -f "${LIB_TBBMALLOC}" ]]; then
   echo "[start_sglang_server_cuda] ERROR: required library not found: ${LIB_TBBMALLOC}" >&2
   exit 1
fi

SGLANG_REQUIRE_IOMP="${SGLANG_REQUIRE_IOMP:-0}"

_find_libiomp() {
   local cand=""

   # 0) explicit override
   if [[ -n "${SGLANG_LIB_IOMP:-}" && -f "${SGLANG_LIB_IOMP}" ]]; then
      echo "${SGLANG_LIB_IOMP}"
      return 0
   fi

   # 1) active env
   for cand in "${PY_PREFIX}/lib/libiomp5.so" "${PY_PREFIX}/lib/libiomp5.so.5"; do
      if [[ -f "${cand}" ]]; then
         echo "${cand}"
         return 0
      fi
   done

   # 2) system locations
   for cand in "/usr/lib/x86_64-linux-gnu/libiomp5.so" "/usr/lib/x86_64-linux-gnu/libiomp5.so.5"; do
      if [[ -f "${cand}" ]]; then
         echo "${cand}"
         return 0
      fi
   done

   # 3) try ldconfig (best-effort)
   if command -v ldconfig >/dev/null 2>&1; then
      cand="$(ldconfig -p 2>/dev/null | awk '/libiomp5\\.so/{print $NF; exit}')" || true
      if [[ -n "${cand}" && -f "${cand}" ]]; then
         echo "${cand}"
         return 0
      fi
   fi

   # 4) search other conda envs under the same conda root (common on remote)
   local envs_dir conda_root
   envs_dir="$(dirname "${PY_PREFIX}")"  # .../envs
   conda_root="$(dirname "${envs_dir}")" # .../
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
      echo "[start_sglang_server_cuda] ERROR: libiomp5.so not found (set SGLANG_LIB_IOMP or install intel-openmp)" >&2
      exit 1
   fi
   echo "[start_sglang_server_cuda] WARN: libiomp5.so not found; continuing without it" >&2
   export LD_PRELOAD="${LIB_TCMALLOC}:${LIB_TBBMALLOC}${LD_PRELOAD:+:${LD_PRELOAD}}"
else
   echo "[start_sglang_server_cuda] Using libiomp: ${LIB_IOMP}" >&2
   export LD_PRELOAD="${LIB_TCMALLOC}:${LIB_TBBMALLOC}:${LIB_IOMP}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

_preload_err="$(LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${LD_PRELOAD}" /usr/bin/true 2>&1)" || {
   echo "[start_sglang_server_cuda] ERROR: LD_PRELOAD test failed" >&2
   echo "${_preload_err}" >&2
   exit 1
}
if [[ -n "${_preload_err}" ]]; then
   echo "[start_sglang_server_cuda] ERROR: LD_PRELOAD produced loader output" >&2
   echo "${_preload_err}" >&2
   exit 1
fi


# ===== Batch Size =====
BATCH_SIZE=${BATCH_SIZE:-16}
echo "Batch size = $BATCH_SIZE"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}

# ===== 绑核与启动 =====
# numactl -C 0-15 \
"${PYTHON_CMD[@]}" -m sglang.launch_server \
   --model-path "$MODEL_DIR" \
   --tokenizer-path "$MODEL_DIR" \
   --trust-remote-code \
   --disable-overlap-schedule \
   --is-embedding \
   --device cuda \
   --host "$HOST" --port "$PORT" \
   --skip-server-warmup \
   --tp 1 \
   --torch-compile-max-bs "$BATCH_SIZE" \
   --log-level error \
   --enable-tokenizer-batch-encode \
   --enable-multimodal \
   --attention-backend triton --sampling-backend pytorch
