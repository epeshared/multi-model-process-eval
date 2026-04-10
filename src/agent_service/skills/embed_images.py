"""embed_images — run image embedding via run_image_embedding.py."""
from __future__ import annotations

from typing import Any, Dict, List

from ..util import REPO_ROOT, run_cmd, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    model_id = str(args.get("model_id") or "").strip()
    base_url = str(args.get("base_url") or "").strip()
    images_dir = str(args.get("images_dir") or "").strip()
    if not model_id:
        raise ValueError("model_id is required")
    if not base_url:
        raise ValueError("base_url is required")
    if not images_dir:
        raise ValueError("images_dir is required")

    argv: List[str] = [
        "python", "scripts/vl-embedding/run_image_embedding.py",
        "--model-id", model_id,
        "--base-url", base_url,
        "--images-dir", images_dir,
    ]

    for key, flag in [
        ("image_size", "--image-size"),
        ("backend", "--backend"),
    ]:
        val = args.get(key)
        if val is not None and str(val).strip():
            argv.extend([flag, str(val).strip()])

    for key, flag in [
        ("batch_size", "--batch-size"),
        ("warmup_samples", "--warmup-samples"),
    ]:
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(int(val))])

    if args.get("normalize"):
        argv.append("--normalize")

    timeout_s = float(args.get("timeout_sec") or 600)
    rc, out = run_cmd(argv, cwd=REPO_ROOT, timeout_s=timeout_s)
    return {"returncode": rc, "command": argv, "output": truncate(out, limit=40000)}


SPEC = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string", "description": "Served model name on the server."},
        "base_url": {"type": "string", "description": "HTTP server URL (required)."},
        "images_dir": {"type": "string", "description": "Directory containing images to embed."},
        "image_size": {"type": "string", "description": "Size filter tag, e.g. '512x512'."},
        "backend": {"type": "string", "enum": ["sglang", "vllm-http"], "default": "sglang"},
        "batch_size": {"type": "integer", "default": 32},
        "warmup_samples": {"type": "integer"},
        "normalize": {"type": "boolean", "default": False},
        "timeout_sec": {"type": "number", "default": 600},
    },
    "required": ["model_id", "base_url", "images_dir"],
}
