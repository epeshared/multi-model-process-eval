"""vl_chat — run vision-language benchmarks via run_vl.py."""
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
        "python", "scripts/vl/run_vl.py",
        "--model", model,
        "--backend", backend,
        "--dataset", dataset,
    ]

    for key, flag in [
        ("model_id", "--model-id"),
        ("base_url", "--base-url"),
        ("device", "--device"),
        ("image_transport", "--image-transport"),
        ("output_jsonl", "--output-jsonl"),
    ]:
        val = args.get(key)
        if val is not None and str(val).strip():
            argv.extend([flag, str(val).strip()])

    for key, flag in [
        ("batch_size", "--batch-size"),
        ("warmup", "--warmup"),
        ("max_new_tokens", "--max-new-tokens"),
        ("synthetic_num_images", "--synthetic-num-images"),
        ("synthetic_image_size", "--synthetic-image-size"),
    ]:
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(int(val))])

    if args.get("profile"):
        argv.append("--profile")

    timeout_s = float(args.get("timeout_sec") or 600)
    rc, out = run_cmd(argv, cwd=REPO_ROOT, timeout_s=timeout_s)
    return {"returncode": rc, "command": argv, "output": truncate(out, limit=40000)}


SPEC = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "enum": ["qwen2.5-vl-3b-instruct", "qwen2.5-vl-7b-instruct"],
            "description": "Vision-language model.",
        },
        "backend": {
            "type": "string",
            "enum": ["torch", "sglang", "sglang-offline", "vllm", "vllm-http"],
            "description": "Inference backend.",
        },
        "dataset": {
            "type": "string",
            "enum": ["single", "flickr8k", "synthetic"],
            "default": "single",
        },
        "model_id": {"type": "string"},
        "base_url": {"type": "string", "description": "HTTP server URL."},
        "device": {"type": "string"},
        "batch_size": {"type": "integer"},
        "warmup": {"type": "integer"},
        "max_new_tokens": {"type": "integer"},
        "image_transport": {"type": "string", "enum": ["data-url", "base64", "path"]},
        "synthetic_num_images": {"type": "integer"},
        "synthetic_image_size": {"type": "integer"},
        "profile": {"type": "boolean", "default": False},
        "output_jsonl": {"type": "string", "description": "Path to save per-sample outputs."},
        "timeout_sec": {"type": "number", "default": 600},
    },
    "required": ["model", "backend"],
}
