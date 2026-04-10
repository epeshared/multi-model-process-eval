"""run_mteb — evaluate embeddings on MTEB benchmarks via run_mteb.py."""
from __future__ import annotations

from typing import Any, Dict, List

from ..util import REPO_ROOT, run_cmd, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    backend = str(args.get("backend") or "").strip()
    model_id = str(args.get("model_id") or "").strip()
    base_url = str(args.get("base_url") or "").strip()
    if not backend:
        raise ValueError("backend is required")
    if not model_id:
        raise ValueError("model_id is required")
    if not base_url:
        raise ValueError("base_url is required")

    argv: List[str] = [
        "python", "scripts/embedding/mteb/run_mteb.py",
        "--backend", backend,
        "--model-id", model_id,
        "--base-url", base_url,
    ]

    for key, flag in [
        ("api", "--api"),
        ("tasks", "--tasks"),
        ("output_folder", "--output-folder"),
    ]:
        val = args.get(key)
        if val is not None and str(val).strip():
            argv.extend([flag, str(val).strip()])

    if args.get("overwrite"):
        argv.append("--overwrite")
    if args.get("clear_cache"):
        argv.append("--clear-cache")

    timeout_s = float(args.get("timeout_sec") or 1800)
    rc, out = run_cmd(argv, cwd=REPO_ROOT, timeout_s=timeout_s)
    return {"returncode": rc, "command": argv, "output": truncate(out, limit=40000)}


SPEC = {
    "type": "object",
    "properties": {
        "backend": {
            "type": "string",
            "enum": ["sglang", "vllm-http", "vllm"],
            "description": "Embedding backend for MTEB.",
        },
        "model_id": {"type": "string", "description": "Model identifier on the server."},
        "base_url": {"type": "string", "description": "HTTP server URL."},
        "api": {"type": "string", "default": "v1", "description": "API mode: v1 or openai."},
        "tasks": {
            "type": "string",
            "description": "Comma-separated MTEB task names (e.g. STSBenchmark,STS14).",
        },
        "output_folder": {"type": "string", "description": "Where to save results."},
        "overwrite": {"type": "boolean", "default": False},
        "clear_cache": {"type": "boolean", "default": False},
        "timeout_sec": {"type": "number", "default": 1800, "description": "Timeout (MTEB can be slow)."},
    },
    "required": ["backend", "model_id", "base_url"],
}
