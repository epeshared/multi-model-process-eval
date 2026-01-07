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

BATCH_SIZE=16
echo "Batch size = $BATCH_SIZE"

python -m sglang.launch_server \
   --model-path "$MODEL_DIR" \
   --tokenizer-path "$MODEL_DIR" \
   --trust-remote-code \
   --disable-overlap-schedule \
   --device cuda \
   --host 0.0.0.0 --port 30000 \
   --skip-server-warmup \
   --tp 1 \
   --torch-compile-max-bs "$BATCH_SIZE" \
   --log-level error \
   --enable-tokenizer-batch-encode \
   --enable-multimodal \
   --attention-backend triton --sampling-backend pytorch
