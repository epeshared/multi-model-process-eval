#!/usr/bin/env bash
set -euo pipefail

# SGLang server for Qwen3 (CUDA)
# Usage:
#   MODEL_DIR=/path/to/Qwen3-1.7B ./start_sglang_server_cuda.sh

WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
MODEL_DIR=${MODEL_DIR:-"/mnt/nvme2n1p1/xtang/models/Qwen/Qwen3-1.7B"}
###############################################
echo "Using model: $MODEL_DIR"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-65536}

echo "Batch size = $BATCH_SIZE"

python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --tokenizer-path "$MODEL_DIR" \
  --trust-remote-code \
  --disable-overlap-schedule \
  --device cuda \
  --host "$HOST" --port "$PORT" \
  --skip-server-warmup \
  --tp "$TP" \
  --torch-compile-max-bs "$BATCH_SIZE" \
  --log-level error \
  --enable-tokenizer-batch-encode \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --attention-backend triton --sampling-backend pytorch
