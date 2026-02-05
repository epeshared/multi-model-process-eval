#!/usr/bin/env bash
set -euo pipefail

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
# MODEL_DIR="/home/xtang/models/openai/clip-vit-base-patch32"
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-large-patch14-336"
# MODEL_DIR="/home/xtang/models/Qwen/Qwen3-Embedding-4B"
# MODEL_DIR="/home/xtang/models/Qwen/Qwen3-Embedding-0.6B"
MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/tencent/youtu-embedding-fp16"}
###############################################
echo "Using model: $MODEL_DIR"

# ===== OneDNN / IPEX 建议 =====
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export DNNL_VERBOSE=0
export IPEX_DISABLE_AUTOCAST=1   # 建议开启，规避 uint64 copy_kernel 坑

# ===== 日志目录 =====
mkdir -p "sglang_logs/sglang_cpu"
export SGLANG_TORCH_PROFILER_DIR="$PWD/sglang_logs/sglang_cpu"

# ===== WORK_HOME 更稳的写法 =====
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

# ===== 环境路径 =====
export SGLANG_USE_CPU_ENGINE=1

# Use the active Python environment.
# - If you want to force a specific interpreter, set SGLANG_PYTHON.
# - Avoid relying on CONDA_PREFIX here (it can point to conda *base*).
PYTHON_BIN="${SGLANG_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found on PATH" >&2
  exit 127
fi
echo "PYTHON_BIN=${PYTHON_BIN}"

PY_PREFIX="$("${PYTHON_BIN}" -c 'import sys; from pathlib import Path; p=Path(sys.executable).resolve(); print(p.parents[1])' 2>/dev/null || true)"
echo "PY_PREFIX=${PY_PREFIX}"

# ===== 预装库（required; exit if not loadable）=====
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

LIB_TCMALLOC="${LD_LIBRARY_PATH}/libtcmalloc.so.4"
LIB_TBBMALLOC="${LD_LIBRARY_PATH}/libtbbmalloc.so.2"
LIB_IOMP="${PY_PREFIX}/lib/libiomp5.so"

for f in "${LIB_TCMALLOC}" "${LIB_TBBMALLOC}" "${LIB_IOMP}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[start_sglang_server] ERROR: required library not found: ${f}" >&2
    exit 1
  fi
done

export LD_PRELOAD="${LIB_TCMALLOC}:${LIB_TBBMALLOC}:${LIB_IOMP}${LD_PRELOAD:+:${LD_PRELOAD}}"

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

# ===== 线程/NUMA（按需调整）=====
export MALLOC_ARENA_MAX=1

# Optional: if you suspect PyTorch/libnuma issues, you can disable PyTorch's
# internal NUMA logic by exporting `C10_DISABLE_NUMA=1` before launch.

# ===== Batch Size (controls --torch-compile-max-bs) =====
# Prefer BATCH_SIZE; allow legacy SERVER_BATCH_SIZE as a fallback.
BATCH_SIZE=${BATCH_SIZE:-${SERVER_BATCH_SIZE:-16}}
echo "Batch size = $BATCH_SIZE"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}

# ===== Memory / cache caps (highly recommended for multi-instance runs) =====
# If unset, sglang sizes KV cache based on perceived available memory, which
# can massively over-allocate when multiple servers start in parallel.
#
# NOTE: This repo previously used SGLANG_MAX_TOTAL_NUM_TOKENS / SGLANG_CONTEXT_LEN.
# The installed SGLang version expects:
#   --max-total-tokens
#   --context-length
SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-${SGLANG_MAX_TOTAL_NUM_TOKENS:-}}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-${SGLANG_CONTEXT_LEN:-}}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-}
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-}
SGLANG_DISABLE_RADIX_CACHE=${SGLANG_DISABLE_RADIX_CACHE:-}

EXTRA_ARGS=()
if [[ -n "${SGLANG_MAX_TOTAL_TOKENS}" ]]; then
  EXTRA_ARGS+=(--max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}")
fi
if [[ -n "${SGLANG_CONTEXT_LENGTH}" ]]; then
  EXTRA_ARGS+=(--context-length "${SGLANG_CONTEXT_LENGTH}")
fi
if [[ -n "${SGLANG_MEM_FRACTION_STATIC}" ]]; then
  EXTRA_ARGS+=(--mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}")
fi
if [[ -n "${SGLANG_CHUNKED_PREFILL_SIZE}" ]]; then
  EXTRA_ARGS+=(--chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}")
fi
if [[ -n "${SGLANG_MAX_PREFILL_TOKENS}" ]]; then
  EXTRA_ARGS+=(--max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}")
fi

case "${SGLANG_DISABLE_RADIX_CACHE}" in
  1|true|TRUE|yes|YES|on|ON)
    EXTRA_ARGS+=(--disable-radix-cache)
    ;;
esac

# Embedding workloads don't require multimodal. Keep it opt-in.
SGLANG_ENABLE_MULTIMODAL=${SGLANG_ENABLE_MULTIMODAL:-0}
if [[ "${SGLANG_ENABLE_MULTIMODAL}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-multimodal)
fi

# Optional: pass an explicit NUMA node hint to sglang.
# This is useful when the process is started under numactl and sglang/torch
# would otherwise try to initialize with an invalid node (e.g. -1).
SGLANG_NUMA_NODE=${SGLANG_NUMA_NODE:-}
NUMA_ARGS=()
if [[ -n "${SGLANG_NUMA_NODE}" ]]; then
  NUMA_ARGS=(--numa-node "${SGLANG_NUMA_NODE}")
fi

# ===== 绑核与启动 =====
"$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --tokenizer-path "$MODEL_DIR" \
  --trust-remote-code \
  --disable-overlap-schedule \
  --is-embedding \
  --device cpu \
  --host "$HOST" --port "$PORT" \
  --skip-server-warmup \
  "${NUMA_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --tp 1 \
  --dtype bfloat16 \
  --enable-torch-compile \
  --torch-compile-max-bs "$BATCH_SIZE" \
  --attention-backend intel_amx \
  --enable-tokenizer-batch-encode \
  --log-level info

# numactl -C 0-15 \
# python -m sglang.launch_server \
#   --model-path "$MODEL_DIR" \
#   --tokenizer-path "$MODEL_DIR" \
#   --trust-remote-code \
#   --disable-overlap-schedule \
#   --is-embedding \
#   --device cuda \
#   --host 0.0.0.0 --port 30000 \
#   --skip-server-warmup \
#   --tp 1 \
#   --enable-torch-compile \
#   --torch-compile-max-bs "$BATCH_SIZE" \
#   --log-level error