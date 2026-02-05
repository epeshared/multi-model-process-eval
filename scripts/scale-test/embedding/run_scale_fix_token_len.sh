#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CFG="${SCRIPT_DIR}/config_scale_fix_token_len.json"

# Accept either:
#   ./run_scale_fix_token_len.sh --tee
#   ./run_scale_fix_token_len.sh --config path/to/config.json --tee
#   ./run_scale_fix_token_len.sh path/to/config.json --tee   (back-compat)
if [[ ${1:-} == "--config" ]]; then
	if [[ -z ${2:-} ]]; then
		echo "error: --config requires a value" >&2
		exit 2
	fi
	CFG="$2"
	shift 2
elif [[ ${1:-} != "" && ${1:-} != -* ]]; then
	CFG="$1"
	shift || true
fi

python "${SCRIPT_DIR}/run_scale_fix_token_len.py" --config "$CFG" "$@"
