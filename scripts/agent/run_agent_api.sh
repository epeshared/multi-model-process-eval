#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   pip install -r requirements-agent.txt
#   AGENT_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
#   AGENT_OPENAI_API_KEY=... \
#   AGENT_MODEL=... \
#   bash scripts/agent/run_agent_api.sh

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8899}

exec python3 -m uvicorn src.agent_service.app:app --host "$HOST" --port "$PORT"
