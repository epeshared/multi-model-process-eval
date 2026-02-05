#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-scripts/scale-test/embedding/config_scale_fix_token_len.json}
shift || true

python3 scripts/scale-test/embedding/run_scale_fix_token_len.py --config "$CFG" "$@"
