#!/usr/bin/env bash
set -euo pipefail

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-base-patch32"
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-large-patch14-336"
MODEL_DIR="/home/xtang//models/Qwen/Qwen3-Embedding-4B"
# MODEL_DIR="$WORK_HOME/models/Qwen/Qwen3-Embedding-0.6B"
###############################################
echo "Using model: $MODEL_DIR"



# ===== Batch Size =====
BATCH_SIZE=16
echo "Batch size = $BATCH_SIZE"

# ===== 绑核与启动 =====
# numactl -C 0-15 \
python -m sglang.launch_server \
   --model-path "$MODEL_DIR" \
   --tokenizer-path "$MODEL_DIR" \
   --trust-remote-code \
   --disable-overlap-schedule \
   --is-embedding \
   --device cuda \
   --host 0.0.0.0 --port 30000 \
   --skip-server-warmup \
   --tp 1 \
   --torch-compile-max-bs "$BATCH_SIZE" \
   --log-level error \
   --enable-tokenizer-batch-encode \
   --enable-multimodal \
   --attention-backend triton --sampling-backend pytorch
