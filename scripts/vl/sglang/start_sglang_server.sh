#!/usr/bin/env bash
set -euo pipefail

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
# MODEL_DIR="/home/xtang/models/openai/clip-vit-base-patch32"
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-large-patch14-336"
# MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-VL-7B-Instruct"}
MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-VL-3B-Instruct"}
# MODEL_DIR="/home/xtang/models/Qwen/Qwen3-Embedding-0.6B"
# MODEL_DIR="/home/xtang/models/Qwen/Qwen3-Embedding-4B"
###############################################
echo "Using model: $MODEL_DIR"

# ===== Serve mode =====
# This script is intended for VL chat. For embedding-only servers, use the embedding scripts.
DEVICE=${DEVICE:-cpu}
ENABLE_MULTIMODAL=${ENABLE_MULTIMODAL:-1}

case "${ENABLE_MULTIMODAL}" in
  0|false|FALSE|no|NO|off|OFF)
    ENABLE_MULTIMODAL=0
    ;;
  *)
    ENABLE_MULTIMODAL=1
    ;;
esac

# if [[ "${DEVICE}" == "cpu" ]] && [[ "${ENABLE_MULTIMODAL}" == "1" ]]; then
#   echo "[start_sglang_server] ERROR: VL multimodal serving on CPU is not supported in this env." >&2
#   echo "[start_sglang_server] The server hit: 'Torch not compiled with CUDA enabled'." >&2
#   echo "[start_sglang_server] Fix: start a CUDA SGLang server (use start_sglang_server_cuda.sh)" >&2
#   echo "[start_sglang_server]      or run client with BACKEND=torch for CPU inference." >&2
#   exit 2
# fi

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
if [[ "${DEVICE}" == "cpu" ]]; then
  export SGLANG_USE_CPU_ENGINE=1
fi

# Use the active Python environment.
# - If you want to force a specific interpreter, set SGLANG_PYTHON.
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

PY_PREFIX="$(${PYTHON_BIN} -c 'import sys; from pathlib import Path; p=Path(sys.executable).resolve(); print(p.parents[1])' 2>/dev/null || true)"
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

# ===== Batch Size =====
BATCH_SIZE=16
echo "Batch size = $BATCH_SIZE"

# ===== 绑核与启动 =====
"$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --tokenizer-path "$MODEL_DIR" \
  --trust-remote-code \
  --disable-overlap-schedule \
  $( [[ "${ENABLE_MULTIMODAL}" == "1" ]] && echo --enable-multimodal ) \
  --device "${DEVICE}" \
  --host 0.0.0.0 --port 30000 \
  --skip-server-warmup \
  --tp 1 \
  --enable-torch-compile \
  --torch-compile-max-bs "$BATCH_SIZE" \
  --attention-backend intel_amx \
  --enable-tokenizer-batch-encode \
  --log-level error \
  --disable-fast-image-processor
  

# --disable-fast-image-processor \  

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
