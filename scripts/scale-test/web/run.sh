#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"

PORT=${1:-30200}
exec python3 scripts/scale-test/web/server.py --port "$PORT"
