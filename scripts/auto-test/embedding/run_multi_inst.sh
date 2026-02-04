#!/usr/bin/env bash
set -euo pipefail

# Run multiple auto-test configs in parallel, with:
# - per-config stdout/stderr logs (avoid interleaved --tee output)
# - Ctrl+C / exit cleanup
# - run all configs in parallel

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

JOB_NAME=${JOB_NAME:-fix_token_len_20_qwen3-embedding-4b_sglang-online}

# Set to 1 to also stream each run into the terminal.
# NOTE: running with --tee across multiple instances will interleave output.
TEE=${TEE:-1}

# By default we run in "legacy" mode for best throughput (no cache caps).
# If you need OOM protection for many parallel instances, set SAFE_MODE=1.
SAFE_MODE=${SAFE_MODE:-0}
if [[ "${SAFE_MODE}" == "1" ]]; then
	# Multi-instance safety defaults.
	# Each sglang server sizes KV cache based on perceived free memory; when many
	# servers start concurrently it can over-allocate and get OOM-killed.
	# Override as needed (set empty to disable passing the arg).
	export SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-65536}
	export SGLANG_ENABLE_MULTIMODAL=${SGLANG_ENABLE_MULTIMODAL:-0}
else
	# Ensure previous SAFE runs don't leak into throughput runs.
	unset SGLANG_MAX_TOTAL_TOKENS SGLANG_MAX_TOTAL_NUM_TOKENS
	unset SGLANG_CONTEXT_LENGTH SGLANG_CONTEXT_LEN
	unset SGLANG_MEM_FRACTION_STATIC
	unset SGLANG_ENABLE_MULTIMODAL
fi

CONFIGS=(
	config_fix_token_len.json
	config_fix_token_len2.json
	config_fix_token_len3.json
	config_fix_token_len4.json
  config_fix_token_len5.json
  config_fix_token_len6.json
  config_fix_token_len7.json
  config_fix_token_len8.json
)

LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/result/multi_inst_logs"}
mkdir -p "${LOG_DIR}"

PIDS=()
NAMES=()

cleanup() {
	local rc=$?
	if ((${#PIDS[@]} > 0)); then
		echo "[multi] cleaning up ${#PIDS[@]} child processes..." >&2
		for pid in "${PIDS[@]}"; do
			kill -TERM "${pid}" 2>/dev/null || true
		done
		sleep 1
		for pid in "${PIDS[@]}"; do
			kill -KILL "${pid}" 2>/dev/null || true
		done
	fi
	exit "${rc}"
}

trap cleanup INT TERM

start_one() {
	local cfg="$1"
	local base
	base=$(basename "${cfg}")
	base="${base%.json}"

	if [[ ! -f "${cfg}" ]]; then
		echo "[multi] missing config: ${cfg}" >&2
		return 2
	fi

	local ts
	ts=$(date -u +%Y%m%dT%H%M%SZ)
	local out_log="${LOG_DIR}/${ts}_${base}.log"

	echo "[multi] start cfg=${cfg} job=${JOB_NAME} log=${out_log}" >&2

	local cmd=("./run_auto_test.sh" "${cfg}" "--stop-servers-after-job" "--only" "${JOB_NAME}")
	if [[ "${TEE}" == "1" ]]; then
		cmd+=("--tee")
		# Keep output ordered per-process (best-effort)
		stdbuf -oL -eL "${cmd[@]}" 2>&1 | tee "${out_log}" &
	else
		stdbuf -oL -eL "${cmd[@]}" >"${out_log}" 2>&1 &
	fi

	local pid=$!
	PIDS+=("${pid}")
	NAMES+=("${base}")
}

for cfg in "${CONFIGS[@]}"; do
	start_one "${cfg}"
done

fail=0
for i in "${!PIDS[@]}"; do
	pid="${PIDS[$i]}"
	name="${NAMES[$i]}"
	if wait "${pid}"; then
		echo "[multi] done ${name} (pid=${pid})" >&2
	else
		rc=$?
		echo "[multi] FAIL ${name} (pid=${pid} rc=${rc})" >&2
		fail=1
	fi
done

trap - INT TERM
if [[ "${fail}" -ne 0 ]]; then
	exit 1
fi

echo "[multi] all done" >&2
