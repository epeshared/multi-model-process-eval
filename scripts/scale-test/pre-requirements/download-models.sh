#!/usr/bin/env bash
set -euo pipefail
# 从环境变量读取 HF_TOKEN
# : "${HF_TOKEN:?Please set HF_TOKEN env var before running this script}"

export HF_ENDPOINT="https://hf-mirror.com"

MODEL_ROOT="${MODEL_ROOT:-${HOME}/models}"
mkdir -p "${MODEL_ROOT}"

# Keep local-dir paths under MODEL_ROOT so this script works for non-root users
# (e.g. ubuntu on cloud images).
huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir "${MODEL_ROOT}/Qwen/Qwen3-Embedding-4B" --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir "${MODEL_ROOT}/Qwen/Qwen3-Embedding-0.6B" --local-dir-use-symlinks False

# huggingface-cli download xxx/yyy --token "$HF_TOKEN" ...
# huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir models/bge-large-zh-v1.5 --local-dir-use-symlinks False
# huggingface-cli download openai/clip-vit-base-patch32 --local-dir models/openai/clip-vit-base-patch32 --local-dir-use-symlinks False
# huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir models/openai/clip-vit-large-patch14-336 --local-dir-use-symlinks False
# huggingface-cli download --token "$HF_TOKEN" Qwen/Qwen3-Embedding-4B --local-dir models/Qwen/Qwen3-Embedding-4B
# huggingface-cli download openai/clip-vit-base-patch32 --local-dir models/openai/clip-vit-base-patch32 --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir models/Qwen/Qwen3-Embedding-4B --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen/Qwen3-Embedding-0.6B --local-dir-use-symlinks False
# huggingface-cli download C-MTEB/LCQMC --local-dir datasets/C-MTEB/LCQMC --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen/Qwen2.5-VL-7B-Instruct --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models/Qwen/Qwen2.5-VL-3B-Instruct --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir models/Qwen/Qwen2.5-Omni-7B --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen2.5-Omni-3B --local-dir models/Qwen/Qwen2.5-Omni-3B --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen/Qwen3-0.6B --local-dir-use-symlinks False
# huggingface-cli download --token "$HF_TOKEN" lmms-lab/Video-MME --local-dir datasets/lmms-lab/Video-MME --local-dir-use-symlinks False
# huggingface-cli download --token "$HF_TOKEN" Qwen/Qwen3-VL-Embedding-2B --local-dir models/Qwen/Qwen3-VL-Embedding-2B --local-dir-use-symlinks False
# huggingface-cli download --token "$HF_TOKEN" Qwen/Qwen3-VL-Embedding-8B --local-dir models/Qwen/Qwen3-VL-Embedding-8B --local-dir-use-symlinks False