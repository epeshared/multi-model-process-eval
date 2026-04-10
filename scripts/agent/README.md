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

## Skills (20)

### Wiki

| Skill | Description |
|-------|-------------|
| `wiki_search` | Search the project wiki by keyword or regex |
| `wiki_read` | Read a wiki page or list all pages |
| `wiki_ingest` | Create or update a wiki page (+ log/index) |
| `wiki_lint` | Health-check wiki for orphans, broken links, etc. |

### Task Runners

| Skill | Description |
|-------|-------------|
| `embed_texts` | Text/image embedding benchmarks (torch/sglang/vllm) |
| `generate_text` | Qwen3 LLM text generation with TTFT/TPOT |
| `vl_chat` | Vision-language (Qwen2.5-VL) image+text chat |
| `omni_chat` | Omni multimodal (Qwen2.5-Omni) benchmarks |
| `embed_images` | Image-only embedding via HTTP server |
| `run_mteb` | MTEB standardized embedding evaluation |
| `dequantize_model` | Convert FP8-quantized weights to FP16/BF16 |
| `auto_test` | Automated multi-config tests with server lifecycle |

### Scale-Test

| Skill | Description |
|-------|-------------|
| `scale_run_fix_token_len` | Launch/resume scale-test sweep |
| `scale_status_fix_token_len` | Query run progress (aggregate.csv, per-host) |
| `remote_preflight_fix_token_len` | SSH preflight checks for remote hosts |
| `scale_analyze` | Post-hoc analysis: pivot tables, plots, EMON extraction |
| `scale_monitor` | One-shot status poll of a running sweep |
| `scale_web_server` | Start/stop scale-test results web UI |
| `gen_test_images` | Generate synthetic test images for VL-embedding |
| `log_analyze` | Rule-based log failure pattern detection |
