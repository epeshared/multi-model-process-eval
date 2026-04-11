---
title: Agent Skills Reference
created: 2026-04-11
updated: 2026-04-11
tags: [guide, agent, skills, api]
sources: [src/agent_service/skills/, src/agent_service/app.py, src/agent_service/config.py]
---

# Agent Skills Reference

Complete reference for all 20 agent skills. Invoke via `POST /v1/skills/{name}` or through the agentic chat loop at `POST /v1/agent/chat`.

## Agent Service Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_OPENAI_BASE_URL` | `http://127.0.0.1:8000/v1` | LLM backend URL |
| `AGENT_OPENAI_API_KEY` | `""` | API key for LLM |
| `AGENT_MODEL` | `""` (required) | Model name for tool-call loop |
| `AGENT_MAX_TOOL_STEPS` | `5` (clamped 0–20) | Max tool calls per chat turn |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Health check → `{"ok": true}` |
| `GET` | `/v1/skills` | List all registered skills with schemas |
| `POST` | `/v1/skills/{name}` | Invoke one skill: `{"args": {...}}` |
| `POST` | `/v1/agent/chat` | Agentic loop: `{"messages": [...], "enable_tools": true}` |

## Wiki Skills

### wiki_search

Search wiki pages by keyword or regex.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | — | Search keyword or regex pattern |
| `scope` | string | no | `all` | Filter: `all`, `index`, `entities`, `concepts`, `guides`, `comparisons`, `sources` |

**Returns:** Matching lines with file paths.

### wiki_read

Read a wiki page or list all pages.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `page` | string | no | — | Page path (e.g. `concepts/amx.md`). Omit to list all pages. |

**Returns:** Page content or directory listing.

### wiki_ingest

Create or update a wiki page.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `page` | string | yes | — | Target page path |
| `content` | string | yes | — | Full markdown content |
| `log_entry` | string | no | — | Append to `wiki/log.md` |
| `index_row` | string | no | — | Append row to `wiki/index.md` |

### wiki_lint

Health-check the wiki.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `checks` | array | no | all | Which checks: `orphans`, `broken_links`, `missing_frontmatter`, `empty_pages` |

**Returns:** List of issues found per check type.

## Task Runner Skills

### embed_texts

Run text/image embedding benchmarks.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | — | Model name or path |
| `dataset` | string | yes | — | Dataset: `yahoo`, `flickr8k`, `synthetic`, `custom` |
| `backend` | string | no | `torch` | `torch`, `sglang`, `vllm`, `vllm-http` |
| `batch_size` | int | no | 32 | Batch size |
| `max_samples` | int | no | 10000 | Max samples |
| `warmup` | int | no | 1 | Warmup samples |
| `timeout` | float | no | 900 | Timeout (sec) |
| `profile` | bool | no | false | Enable torch profiler |
| `use_amx` | bool | no | true | Enable AMX acceleration |
| `synthetic_token_len` | int | no | — | Fixed token length for synthetic mode |
| `max_length` | int | no | — | Tokenizer max length |
| `dtype` | string | no | `bfloat16` | Model dtype |
| `normalize` | bool | no | true | L2 normalize embeddings |

### generate_text

Run Qwen3 LLM text generation.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | — | Model name |
| `backend` | string | yes | — | `sglang` or `vllm-http` |
| `dataset` | string | no | `synthetic` | `single` or `synthetic` |
| `max_new_tokens` | int | no | 256 | Max generation tokens |
| `batch_size` | int | no | 1 | Concurrent requests |
| `warmup` | int | no | 1 | Warmup count |
| `stream` | bool | no | true | Enable streaming (for TTFT/TPOT) |
| `prompt` | string | no | — | Single prompt text |
| `synthetic_num_prompts` | int | no | 100 | Number of synthetic prompts |

### vl_chat

Run vision-language (Qwen2.5-VL) benchmarks.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | — | Model name |
| `backend` | string | yes | — | Any supported backend |
| `dataset` | string | no | `synthetic` | `single`, `flickr8k`, or `synthetic` |
| `batch_size` | int | no | 1 | Batch size |
| `warmup` | int | no | 1 | Warmup count |
| `max_new_tokens` | int | no | 256 | Max generation tokens |
| `image_transport` | string | no | `data-url` | Image encoding mode |
| `synthetic_num_images` | int | no | 10 | Synthetic image count |
| `profile` | bool | no | false | Enable SGLang profiling |

### omni_chat

Run Omni multimodal benchmarks.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | — | Model name |
| `backend` | string | yes | — | `sglang` or `vllm-http` |
| `dataset` | string | no | `synthetic` | Only `synthetic` supported |
| `batch_size` | int | no | 1 | Batch size |
| `warmup` | int | no | 1 | Warmup count |
| `max_new_tokens` | int | no | 256 | Max generation tokens |
| `synthetic_num_images` | int | no | 10 | Synthetic image count |
| `profile` | bool | no | false | Enable profiling |

### embed_images

Run image-only embedding via HTTP server.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_id` | string | yes | — | Served model name |
| `base_url` | string | yes | — | Server URL |
| `images_dir` | string | yes | — | Image directory path |
| `backend` | string | no | `sglang` | `sglang` or `vllm-http` |
| `batch_size` | int | no | 32 | Batch size |
| `warmup_samples` | int | no | 1 | Warmup count |
| `image_size` | string | no | — | Filter tag (e.g. `512x512`) |
| `normalize` | bool | no | true | L2 normalize |

### run_mteb

Evaluate embeddings on MTEB benchmarks.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `backend` | string | yes | — | Backend type |
| `model_id` | string | yes | — | Served model name |
| `base_url` | string | yes | — | Server URL |
| `api` | string | no | `v1` | API style |
| `tasks` | string | no | — | Comma-separated MTEB tasks |
| `output_folder` | string | no | — | Output directory |
| `overwrite` | bool | no | false | Overwrite existing results |
| `clear_cache` | bool | no | false | Clear MTEB cache |
| `timeout_sec` | float | no | 1800 | Process timeout |

### dequantize_model

Convert FP8-quantized model weights to FP16/BF16.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `in_model_dir` | string | yes | — | Input FP8 model directory |
| `out_model_dir` | string | yes | — | Output directory |
| `dtype` | string | no | `float16` | Target dtype: `float16` or `bfloat16` |
| `overwrite` | bool | no | false | Overwrite existing output |
| `keep_quant_aux` | bool | no | false | Keep quantization auxiliary files |
| `verbose` | bool | no | false | Verbose logging |
| `timeout_sec` | float | no | 1800 | Process timeout |

### auto_test

Orchestrate automated multi-config tests.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `config_path` | string | yes | — | Config JSON path |
| `only` | array | no | — | Filter job names |
| `skip` | array | no | — | Skip job names |
| `dry_run` | bool | no | false | Print without executing |
| `tee` | bool | no | false | Stream to terminal |
| `restart_servers` | bool | no | false | Force server restart |
| `stop_servers_after_job` | bool | no | false | Stop servers between jobs |
| `reparse_run_id` | string | no | — | Re-parse old run logs |
| `timeout_sec` | float | no | 7200 | Process timeout |

## Scale-Test Skills

### scale_run_fix_token_len

Launch or resume a scale-test sweep.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `config_path` | string | yes | — | Config JSON path |
| `scale_id` | string | no | — | Fixed run ID |
| `resume` | bool | no | false | Skip completed work |
| `tee` | bool | no | false | Stream output |
| `dry_run` | bool | no | false | Parse only |
| `extra_args` | array | no | `[]` | Forwarded to runner |

### scale_status_fix_token_len

Query run progress.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `scale_id` | string | yes | — | Run ID to check |
| `config_path` | string | no | — | Infer result_root from config |
| `result_root` | string | no | — | Override result root |

**Returns:** `{scale_id, run_dir, exists, aggregate: {counts}, hosts: [{host, aggregate, remote_run_log}]}`

### scale_analyze

Post-hoc analysis on completed runs.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `run_dir` | string | yes | — | Run directory with aggregate.csv |
| `out_dir` | string | no | `analysis` | Output subdirectory name |
| `socket_metrics` | array | no | — | EMON socket-view metric keys |

**Returns:** `{generated_files: [...], output: "..."}`

### scale_monitor

Poll status of a running sweep (single snapshot).

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `scale_id` | string | yes | — | Run ID |
| `task` | string | no | `embedding` | `embedding` or `vl-embedding` |
| `result_root` | string | no | — | Override result root |

### scale_web_server

Manage scale-test results web UI.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `action` | string | no | `start` | `start`, `stop`, or `status` |
| `port` | int | no | 8080 | HTTP port |
| `host` | string | no | `0.0.0.0` | Bind address |

### gen_test_images

Generate synthetic test images.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `out_dir` | string | yes | — | Output directory |
| `sizes` | string | no | `224x224,...,1920x1080` | Comma-separated WxH |
| `per_size` | int | no | 4 | Images per resolution |
| `pattern` | string | no | `checker` | `checker`, `gradient`, or `noise` |
| `format` | string | no | `png` | `png` or `jpg` |
| `seed` | int | no | 0 | RNG seed (for noise) |

### remote_preflight_fix_token_len

SSH preflight checks for remote hosts.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `config_path` | string | yes | — | Config JSON with `run.servers` |

**Returns:** `{hosts: [{ip, returncode, output: "[preflight] whoami=...\n..."}]}`

### log_analyze

Rule-based log failure pattern detection.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `paths` | array | yes | — | Log file paths to analyze |

**Detected patterns:** SSH proxy timeout, conda missing, pip failure, permission denied, OOM, CUDA error, SGLang crash.

## Related

- [Running Benchmarks](running-benchmarks.md) — benchmark workflow
- [Remote Deployment](remote-deployment.md) — multi-host testing
- [Auto-Test Framework](auto-test-framework.md) — automated sweep config
