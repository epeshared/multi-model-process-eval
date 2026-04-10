---
title: Project Overview
created: 2026-04-10
updated: 2026-04-10
tags: [overview, architecture]
---

# Project Overview

**multi-model-process-eval** is a unified inference benchmarking framework for evaluating multiple model families across multiple backends on Intel CPU (AMX/AVX512) and CUDA hardware.

## Architecture

The project follows a three-layer design:

### Data Layer (`src/data/`)

Handles input construction and dataset loading:

- **embedding_inputs.py** — generic loader for text/JSONL/image inputs with deduplication
- **flickr8k.py** — Flickr8k image-caption dataset with modality filtering
- **yahoo_answers.py** — Yahoo Answers JSONL with q/a/q+a selection

Also supports synthetic data generation (fixed token-length or character-length) built into the embedding task scripts.

### Task Layer (`src/tasks/`)

Each task has a main entry file and a `*_backends/` directory:

| Task | Entry | Backends |
|------|-------|----------|
| [Embedding](entities/tasks/embedding.md) | `embedding.py` | torch, sglang, sglang-offline, vllm, vllm-http |
| [Qwen3 LLM](entities/tasks/qwen3-llm.md) | `qwen3.py` | sglang, vllm-http |
| [VL](entities/tasks/vl.md) | `vl.py` | torch, sglang, sglang-offline, vllm, vllm-http |
| [Omni](entities/tasks/omni.md) | `omni.py` | sglang, vllm, vllm-http |

**Session pattern**: `load_*_session()` loads the model (expensive, one-time), then `embed()` / `chat_with_session()` runs inference (cheap, measurable).

### Script Layer (`scripts/`)

Shell and Python entry points organized by task:

- `scripts/embedding/` — run_embedding.py, run_fix_token_len.sh, server startup scripts
- `scripts/qwen3/` — run_qwen3.py, benchmark scripts
- `scripts/vl/` — run_vl.py, Flickr8k and synthetic test scripts
- `scripts/omni/` — run_omni.py, synthetic benchmark
- `scripts/auto-test/` — automated multi-config test framework with CPU affinity and Emon support
- `scripts/vl-embedding/` — image-only embedding scripts

### Agent Service (`src/agent_service/`)

A FastAPI-based agent service with:
- Skill registry and plugin system (20 skills)
- Wiki skills: search, read, ingest, lint
- Task runner skills: embed_texts, generate_text, vl_chat, omni_chat, embed_images, run_mteb, dequantize_model, auto_test
- Scale-test skills: scale_run, scale_status, remote_preflight, scale_analyze, scale_monitor, scale_web_server, gen_test_images, log_analyze
- OpenAI-compatible LLM client with tool call loop
- Endpoints: `/v1/skills`, `/v1/agent/chat`

## Key Design Patterns

- **Backend abstraction**: `_backend_tag` attribute on session objects lets upper layers identify backend capabilities
- **Image transport**: unified data-url / base64 / file-path modes across all multimodal backends
- **Profiling opt-in**: task layer controls profiler lifecycle; backends are unaware
- **Environment-driven config**: all scripts configurable via env vars with sensible defaults

## Hardware Focus

Primary optimization target is Intel Xeon with [AMX](concepts/amx.md) acceleration:
- `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`
- `--attention-backend intel_amx`
- `LD_PRELOAD` of tcmalloc + tbbmalloc + libiomp5
- [Torch compile](concepts/torch-compile.md) for static graph optimization

See [CPU Optimization Guide](guides/cpu-optimization.md) for full details.

## Related Pages

- [Backend Feature Matrix](comparisons/backend-feature-matrix.md)
- [Running Benchmarks](guides/running-benchmarks.md)
