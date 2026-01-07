#!/usr/bin/env bash
set -euo pipefail

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen2.5-Omni-7B"}
###############################################
echo "Using model: $MODEL_DIR"

# ===== Serve mode =====
# This script is intended for Omni multimodal chat.
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

# ===== OneDNN / IPEX 建议 =====
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export DNNL_VERBOSE=0
export IPEX_DISABLE_AUTOCAST=1

# ===== 日志目录 =====
mkdir -p "sglang_logs/sglang_cpu"
export SGLANG_TORCH_PROFILER_DIR="$PWD/sglang_logs/sglang_cpu"

# ===== WORK_HOME 更稳的写法 =====
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

# ===== 环境路径 =====
export CONDA_PREFIX="/root/miniforge3/envs/xtang-embedding-cpu"
if [[ "${DEVICE}" == "cpu" ]]; then
  export SGLANG_USE_CPU_ENGINE=1
fi

# ===== 预装库（安全拼接 LD_PRELOAD）=====
LIBS=(
  "$CONDA_PREFIX/lib/libiomp5.so"
  "$CONDA_PREFIX/lib/libtcmalloc.so"
  "$CONDA_PREFIX/lib/libtbbmalloc.so.2"
)
PRELOAD_JOIN=""
for f in "${LIBS[@]}"; do
  [[ -f "$f" ]] && PRELOAD_JOIN="${PRELOAD_JOIN:+$PRELOAD_JOIN:}$f"
done
export LD_PRELOAD="${PRELOAD_JOIN}${LD_PRELOAD:+:$LD_PRELOAD}"

export MALLOC_ARENA_MAX=1

BATCH_SIZE=16
echo "Batch size = $BATCH_SIZE"

python -m sglang.launch_server \
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
