# Agent Service

This repo can be run as a small FastAPI "agent" service with pluggable skills.

## Install

```bash
pip install -r requirements-agent.txt
```

## Run

```bash
AGENT_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
AGENT_OPENAI_API_KEY=... \
AGENT_MODEL=... \
bash scripts/agent/run_agent_api.sh
```

Service defaults to `http://127.0.0.1:8899`.

## Endpoints

- `GET /healthz`
- `GET /v1/skills`
- `POST /v1/skills/{skill_name}`
- `POST /v1/agent/chat` (OpenAI-compatible tool calling loop)

## Skills (initial)

- `scale_run_fix_token_len`
- `scale_status_fix_token_len`
- `remote_preflight_fix_token_len`
- `log_analyze`
