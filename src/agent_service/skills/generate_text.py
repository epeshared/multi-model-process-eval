"""generate_text — run LLM text generation benchmarks via run_qwen3.py."""
from __future__ import annotations

from typing import Any, Dict, List

from ..util import REPO_ROOT, run_cmd, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    model = str(args.get("model") or "").strip()
    backend = str(args.get("backend") or "").strip()
    dataset = str(args.get("dataset") or "single").strip()
    if not model:
        raise ValueError("model is required")
    if not backend:
        raise ValueError("backend is required")

    argv: List[str] = [
        "python", "scripts/qwen3/run_qwen3.py",
        "--model", model,
        "--backend", backend,
        "--dataset", dataset,
    ]

    for key, flag in [
        ("model_id", "--model-id"),
        ("base_url", "--base-url"),
        ("api_key", "--api-key"),
        ("prompt", "--prompt"),
    ]:
        val = args.get(key)
        if val is not None and str(val).strip():
            argv.extend([flag, str(val).strip()])

    for key, flag in [
        ("max_new_tokens", "--max-new-tokens"),
        ("batch_size", "--batch-size"),
        ("warmup", "--warmup"),
        ("synthetic_num_prompts", "--synthetic-num-prompts"),
        ("synthetic_token_len", "--synthetic-token-len"),
    ]:
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(int(val))])

    if args.get("stream") is True:
        argv.append("--stream")
    elif args.get("stream") is False:
        argv.append("--no-stream")

    timeout_s = float(args.get("timeout_sec") or 600)
    rc, out = run_cmd(argv, cwd=REPO_ROOT, timeout_s=timeout_s)
    return {"returncode": rc, "command": argv, "output": truncate(out, limit=40000)}


SPEC = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "enum": ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"],
            "description": "Qwen3 LLM model size.",
        },
        "backend": {
            "type": "string",
            "enum": ["sglang", "vllm-http"],
            "description": "Inference backend (HTTP only).",
        },
        "dataset": {
            "type": "string",
            "enum": ["single", "synthetic"],
            "default": "single",
            "description": "single = one prompt; synthetic = generated prompts.",
        },
        "model_id": {"type": "string", "description": "Override model HF path or served name."},
        "base_url": {"type": "string", "description": "HTTP server URL."},
        "api_key": {"type": "string"},
        "prompt": {"type": "string", "description": "Prompt text (for dataset=single)."},
        "max_new_tokens": {"type": "integer", "default": 128},
        "batch_size": {"type": "integer"},
        "warmup": {"type": "integer"},
        "stream": {"type": "boolean", "description": "Enable streaming (measures TTFT)."},
        "synthetic_num_prompts": {"type": "integer"},
        "synthetic_token_len": {"type": "integer"},
        "timeout_sec": {"type": "number", "default": 600},
    },
    "required": ["model", "backend"],
}
