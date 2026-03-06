#!/usr/bin/env bash
set -euo pipefail
# 从环境变量读取 HF_TOKEN
# : "${HF_TOKEN:?Please set HF_TOKEN env var before running this script}"

export HF_ENDPOINT="https://hf-mirror.com"

MODEL_ROOT="${MODEL_ROOT:-${HOME}/models}"
mkdir -p "${MODEL_ROOT}"

download_one() {
	local repo_id="$1"
	local local_dir="$2"
	echo "[info] downloading model: ${repo_id} -> ${local_dir}"
	# Retries help with flaky networks / HF mirrors.
	local ok=0
	for i in 1 2 3; do
		if huggingface-cli download "${repo_id}" --local-dir "${local_dir}" --local-dir-use-symlinks False; then
			ok=1
			break
		fi
		echo "[warn] huggingface download failed (attempt ${i}/3): ${repo_id}" >&2
		sleep 3
	done
	if [[ "${ok}" != "1" ]]; then
		echo "[error] failed to download after retries: ${repo_id}" >&2
		return 1
	fi
}

infer_qwen_repo_id() {
	# Infer a HuggingFace repo ID from server_template hints.
	# Returns empty string if unknown.
	local model_id="${MMPE_SERVER_TEMPLATE_MODEL_ID:-}"
	local model="${MMPE_SERVER_TEMPLATE_MODEL:-}"
	local model_dir="${MMPE_SERVER_TEMPLATE_MODEL_DIR:-}"

	# If model_id is already a full HF repo id (org/name), trust it.
	if [[ -n "${model_id}" && "${model_id}" == */* ]]; then
		echo "${model_id}"
		return 0
	fi

	# Common case in this repo: Qwen3-Embedding-0.6B / Qwen3-Embedding-4B
	if [[ -n "${model_id}" && "${model_id}" == Qwen3-Embedding-* ]]; then
		echo "Qwen/${model_id}"
		return 0
	fi

	# Try basename of model_dir.
	if [[ -n "${model_dir}" ]]; then
		local bn
		bn="$(basename "${model_dir}")"
		if [[ "${bn}" == Qwen3-Embedding-* ]]; then
			echo "Qwen/${bn}"
			return 0
		fi
	fi

	# Best-effort: map sglang-style lowercase names.
	# e.g. qwen3-embedding-0.6b -> Qwen3-Embedding-0.6B
	if [[ -n "${model}" && "${model}" == qwen3-embedding-* ]]; then
		local suffix="${model#qwen3-embedding-}"
		# Normalize: 0.6b -> 0.6B
		suffix="${suffix%b}B"
		echo "Qwen/Qwen3-Embedding-${suffix}"
		return 0
	fi

	echo ""
}

infer_local_dir() {
	local model_dir="${MMPE_SERVER_TEMPLATE_MODEL_DIR:-}"
	if [[ -n "${model_dir}" ]]; then
		echo "${model_dir}"
		return 0
	fi
	# Default under MODEL_ROOT (legacy behavior)
	echo "${MODEL_ROOT}"
}

# Prefer: only download the model used by the current sweep (server_template).
# Dispatcher exports:
# - MMPE_SERVER_TEMPLATE_MODEL_DIR
# - MMPE_SERVER_TEMPLATE_MODEL
# - MMPE_SERVER_TEMPLATE_MODEL_ID
repo_id="$(infer_qwen_repo_id)"
local_dir="$(infer_local_dir)"

if [[ -n "${repo_id}" ]]; then
	# If server_template specifies a concrete model_dir, use it.
	# Otherwise, store under MODEL_ROOT with a reasonable default.
	if [[ "${local_dir}" == "${MODEL_ROOT}" ]]; then
		# Put under MODEL_ROOT/<org>/<name>
		org="${repo_id%%/*}"
		name="${repo_id#*/}"
		local_dir="${MODEL_ROOT}/${org}/${name}"
	fi
	download_one "${repo_id}" "${local_dir}"
else
	echo "[warn] could not infer model repo from server_template; falling back to legacy downloads" >&2
	# Keep local-dir paths under MODEL_ROOT so this script works for non-root users
	# (e.g. ubuntu on cloud images).
	download_one Qwen/Qwen3-Embedding-4B "${MODEL_ROOT}/Qwen/Qwen3-Embedding-4B"
	download_one Qwen/Qwen3-Embedding-0.6B "${MODEL_ROOT}/Qwen/Qwen3-Embedding-0.6B"
fi

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