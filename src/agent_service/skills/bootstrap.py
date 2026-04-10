from __future__ import annotations

from .registry import SkillSpec, register
from . import scale_run, scale_status, remote_preflight, log_analyze
from . import scale_analyze, scale_monitor, scale_web_server, gen_test_images
from . import wiki_search, wiki_read, wiki_ingest, wiki_lint
from . import embed_texts, generate_text, vl_chat, omni_chat
from . import embed_images, run_mteb, dequantize_model, auto_test


def register_all() -> None:
    # --- Wiki skills ---
    register(
        SkillSpec(
            name="wiki_search",
            description="Search the project wiki by keyword or regex. Returns matching lines with file paths.",
            parameters_schema=wiki_search.SPEC,
            handler=wiki_search.handler,
        )
    )
    register(
        SkillSpec(
            name="wiki_read",
            description="Read a wiki page or list all pages. Use to browse the knowledge base.",
            parameters_schema=wiki_read.SPEC,
            handler=wiki_read.handler,
        )
    )
    register(
        SkillSpec(
            name="wiki_ingest",
            description="Create or update a wiki page, optionally appending to the log and index.",
            parameters_schema=wiki_ingest.SPEC,
            handler=wiki_ingest.handler,
        )
    )
    register(
        SkillSpec(
            name="wiki_lint",
            description="Health-check the wiki for orphan pages, broken links, missing frontmatter, and empty pages.",
            parameters_schema=wiki_lint.SPEC,
            handler=wiki_lint.handler,
        )
    )

    # --- Task runner skills ---
    register(
        SkillSpec(
            name="embed_texts",
            description="Run text/image embedding benchmarks (Yahoo, Flickr8k, synthetic). Supports torch/sglang/vllm backends.",
            parameters_schema=embed_texts.SPEC,
            handler=embed_texts.handler,
        )
    )
    register(
        SkillSpec(
            name="generate_text",
            description="Run Qwen3 LLM text generation benchmarks with TTFT/TPOT metrics. Supports sglang/vllm-http.",
            parameters_schema=generate_text.SPEC,
            handler=generate_text.handler,
        )
    )
    register(
        SkillSpec(
            name="vl_chat",
            description="Run vision-language (Qwen2.5-VL) image+text chat benchmarks. Supports all backends.",
            parameters_schema=vl_chat.SPEC,
            handler=vl_chat.handler,
        )
    )
    register(
        SkillSpec(
            name="omni_chat",
            description="Run Omni multimodal (Qwen2.5-Omni) benchmarks. Supports sglang/vllm backends.",
            parameters_schema=omni_chat.SPEC,
            handler=omni_chat.handler,
        )
    )
    register(
        SkillSpec(
            name="embed_images",
            description="Run image-only embedding via HTTP server (sglang/vllm). Requires running server.",
            parameters_schema=embed_images.SPEC,
            handler=embed_images.handler,
        )
    )
    register(
        SkillSpec(
            name="run_mteb",
            description="Evaluate embeddings on MTEB standardized benchmarks (STS, classification, etc.).",
            parameters_schema=run_mteb.SPEC,
            handler=run_mteb.handler,
        )
    )
    register(
        SkillSpec(
            name="dequantize_model",
            description="Convert FP8-quantized model weights to FP16/BF16 for CPU inference.",
            parameters_schema=dequantize_model.SPEC,
            handler=dequantize_model.handler,
        )
    )
    register(
        SkillSpec(
            name="auto_test",
            description="Orchestrate automated multi-config embedding tests with server lifecycle, NUMA pinning, and CSV reporting.",
            parameters_schema=auto_test.SPEC,
            handler=auto_test.handler,
        )
    )

    # --- Scale-test skills ---
    register(
        SkillSpec(
            name="scale_run_fix_token_len",
            description="Run/resume the embedding scale-test (fix_token_len) via existing runner.",
            parameters_schema=scale_run.SPEC,
            handler=scale_run.handler,
        )
    )
    register(
        SkillSpec(
            name="scale_status_fix_token_len",
            description="Summarize a scale_id status by inspecting <result_root>/<scale_id>/.",
            parameters_schema=scale_status.SPEC,
            handler=scale_status.handler,
        )
    )
    register(
        SkillSpec(
            name="remote_preflight_fix_token_len",
            description="SSH preflight checks for hosts in the scale-test config (conda/repo/result dirs).",
            parameters_schema=remote_preflight.SPEC,
            handler=remote_preflight.handler,
        )
    )
    register(
        SkillSpec(
            name="log_analyze",
            description="Rule-based log analyzer for common failure patterns (ssh timeout/conda missing/etc).",
            parameters_schema=log_analyze.SPEC,
            handler=log_analyze.handler,
        )
    )
    register(
        SkillSpec(
            name="scale_analyze",
            description="Run post-hoc analysis on a completed scale-test run — generates pivot tables, scaling CSVs, and PNG plots.",
            parameters_schema=scale_analyze.SPEC,
            handler=scale_analyze.handler,
        )
    )
    register(
        SkillSpec(
            name="scale_monitor",
            description="Poll status of a running scale-test (progress, per-host breakdown, errors). Captures one snapshot.",
            parameters_schema=scale_monitor.SPEC,
            handler=scale_monitor.handler,
        )
    )
    register(
        SkillSpec(
            name="scale_web_server",
            description="Start/stop/check the scale-test results web UI (lightweight HTTP browser for CSV/PNG analysis).",
            parameters_schema=scale_web_server.SPEC,
            handler=scale_web_server.handler,
        )
    )
    register(
        SkillSpec(
            name="gen_test_images",
            description="Generate synthetic test images of various resolutions for VL-embedding throughput benchmarks.",
            parameters_schema=gen_test_images.SPEC,
            handler=gen_test_images.handler,
        )
    )
