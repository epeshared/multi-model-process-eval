#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

CONFIG_ARG=()

# Convenience: allow passing config JSON as first positional arg.
# Examples:
#   ./run_auto_test.sh config_yahoo.json
#   ./run_auto_test.sh config_mteb.json --tee --only mteb_STSBenchmark_qwen3-embedding-4b_vllm-http
if [[ $# -gt 0 ]] && [[ "${1:-}" != "-"* ]]; then
	CANDIDATE="$1"
	if [[ "${CANDIDATE}" == *.json ]] && [[ -f "${CANDIDATE}" ]]; then
		CONFIG_ARG=(--config "${CANDIDATE}")
		shift
	elif [[ "${CANDIDATE}" == /* ]] && [[ -f "${CANDIDATE}" ]]; then
		CONFIG_ARG=(--config "${CANDIDATE}")
		shift
	fi
fi

python3 "${ROOT_DIR}/scripts/auto-test/embedding/run_auto_test.py" "${CONFIG_ARG[@]}" "$@"
