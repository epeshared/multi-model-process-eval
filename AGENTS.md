# Wiki Schema — multi-model-process-eval

This file defines the conventions for maintaining the project wiki under `wiki/`.
Any LLM agent working on this repository should read this file first.

## Three Layers

| Layer | Path | Owner | Mutability |
|-------|------|-------|------------|
| Raw sources | `raw/` | Human | Immutable — LLM reads, never modifies |
| Wiki | `wiki/` | LLM | LLM creates and maintains all pages |
| Schema | `AGENTS.md` (this file) | Human + LLM | Co-evolved over time |

## Directory Layout

```
raw/
├── sources/          # Articles, papers, vendor docs, release notes
├── results/          # Raw benchmark outputs (CSV, JSON, logs)
└── assets/           # Images, charts, diagrams

wiki/
├── index.md          # Content catalog — updated on every ingest
├── log.md            # Chronological append-only operation log
├── overview.md       # High-level project synthesis
│
├── entities/         # One page per distinct entity
│   ├── models/       # Model family pages (qwen3-embedding.md, clip.md, ...)
│   ├── backends/     # Backend pages (sglang.md, vllm.md, torch.md)
│   └── tasks/        # Task pages (embedding.md, qwen3-llm.md, vl.md, omni.md)
│
├── concepts/         # One page per technical concept
│                     # (ttft.md, tpot.md, amx.md, torch-compile.md, ...)
│
├── guides/           # How-to guides synthesized from experience
│                     # (cpu-optimization.md, multi-instance.md, ...)
│
├── comparisons/      # Cross-cutting comparison pages
│                     # (sglang-vs-vllm-embedding.md, ...)
│
└── sources/          # One summary page per ingested source
```

## Page Conventions

### Frontmatter

Every wiki page starts with YAML frontmatter:

```yaml
---
title: Page Title
created: 2026-04-10
updated: 2026-04-10
tags: [embedding, sglang, benchmark]
sources: [raw/results/embedding_sglang_2026-04-10.csv]
---
```

### Cross-references

Use standard Markdown links to other wiki pages:

```markdown
See [SGLang backend](../backends/sglang.md) for server configuration.
Related: [TTFT](../concepts/ttft.md) | [Torch Compile](../concepts/torch-compile.md)
```

### Heading Structure

- `#` — Page title (matches frontmatter `title`)
- `##` — Major sections
- `###` — Subsections

## Operations

### Ingest

When a new source is added to `raw/`:

1. Read the source document fully.
2. Create a summary page in `wiki/sources/`.
3. Update or create relevant entity/concept pages across the wiki.
4. Update `wiki/index.md` with new entries.
5. Append an entry to `wiki/log.md`.

A single source may touch 5–15 wiki pages.

### Query

When answering a question:

1. Read `wiki/index.md` to find relevant pages.
2. Read those pages for context.
3. Synthesize an answer with citations to wiki pages and raw sources.
4. If the answer is reusable, file it as a new wiki page (guide or comparison).

### Lint

Periodically check for:

- Contradictions between pages
- Stale claims superseded by newer benchmark results
- Orphan pages with no inbound links
- Missing pages for frequently mentioned concepts
- Outdated performance numbers

## Log Format

Each log entry in `wiki/log.md` follows:

```markdown
## [2026-04-10] ingest | Source Title
- Summary: one-line description
- Pages touched: list of updated wiki pages
```

## Index Format

Each entry in `wiki/index.md`:

```markdown
| Page | Category | Summary | Last Updated |
```

Grouped by category (entities, concepts, guides, comparisons, sources).

## Domain Context

This project benchmarks inference performance across:

- **Models**: Qwen3-Embedding (0.6B, 4B), CLIP, Youtu-Embedding, Qwen3 LLM, Qwen2.5-VL, Qwen2.5-Omni
- **Backends**: torch (local), SGLang (offline + HTTP), vLLM (offline + HTTP)
- **Tasks**: text embedding, image embedding, LLM chat, vision-language chat, omni multimodal
- **Hardware focus**: Intel CPU with AMX/AVX512 optimization

Key metrics to track: throughput (samples/sec), latency (ms), TTFT, TPOT, batch size scaling, memory usage.
