#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CFG="${SCRIPT_DIR}/config_scale_fix_token_len_amd.json"

# Accept either:
#   ./run_scale_fix_token_len.sh --tee
#   ./run_scale_fix_token_len.sh --config path/to/config.json --tee
#   ./run_scale_fix_token_len.sh path/to/config.json --tee   (back-compat)
#
# Extra options (forwarded to the Python runner):
#   --remote-clean-repo[=true|false]   If true, delete remote_repo_dir on the remote host
#                                     before rsync syncing this repo (default: false).
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


# Default to streaming output (including remote SSH output) unless the user
# explicitly disables it. This avoids the common "printed remote_run_dir then
# nothing" confusion while remote setup/pip installs are still running.
want_default_tee=1
for a in "$@"; do
		if [[ "$a" == "--tee" || "$a" == "--no-tee" ]]; then
				want_default_tee=0
				break
		fi
done

extra_args=()
if [[ $want_default_tee -eq 1 ]]; then
		extra_args+=("--tee")
fi

python "${SCRIPT_DIR}/run_scale_fix_token_len.py" --config "$CFG" "${extra_args[@]}" "$@"
