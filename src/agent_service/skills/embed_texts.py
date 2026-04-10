"""embed_texts — run text/image embedding benchmarks via run_embedding.py."""
from __future__ import annotations

from typing import Any, Dict, List

from ..util import REPO_ROOT, run_cmd, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    model = str(args.get("model") or "").strip()
    backend = str(args.get("backend") or "torch").strip()
    dataset = str(args.get("dataset") or "").strip()
    if not model:
        raise ValueError("model is required")
    if not dataset:
        raise ValueError("dataset is required")

    argv: List[str] = [
        "python", "scripts/embedding/run_embedding.py",
        "--model", model,
        "--backend", backend,
        "--dataset", dataset,
    ]

    # Optional string args
    for key, flag in [
        ("model_id", "--model-id"),
        ("dataset_path", "--dataset-path"),
        ("yahoo_mode", "--yahoo-mode"),
        ("device", "--device"),
        ("base_url", "--base-url"),
        ("api", "--api"),
        ("api_key", "--api-key"),
        ("dtype", "--dtype"),
        ("encoding_format", "--encoding-format"),
        ("output_path", "--output-path"),
        ("flickr8k_images_dir", "--flickr8k-images-dir"),
        ("flickr8k_captions_file", "--flickr8k-captions-file"),
    ]:
        val = args.get(key)
        if val is not None and str(val).strip():
            argv.extend([flag, str(val).strip()])

    # Optional int args
    for key, flag in [
        ("max_samples", "--max-samples"),
        ("batch_size", "--batch-size"),
        ("warmup_samples", "--warmup-samples"),
        ("max_length", "--max-length"),
        ("tp_size", "--tp-size"),
        ("max_model_len", "--max-model-len"),
        ("synthetic_token_len", "--synthetic-token-len"),
        ("synthetic_input_len", "--synthetic-input-len"),
    ]:
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(int(val))])

    # Optional float args
    for key, flag in [
        ("timeout", "--timeout"),
        ("gpu_memory_utilization", "--gpu-memory-utilization"),
    ]:
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(float(val))])

    # Boolean flags
    if args.get("normalize") is False:
        argv.append("--no-normalize")
    if args.get("use_amx"):
        argv.append("--use-amx")
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
            "enum": ["qwen3-embedding-4b", "qwen3-embedding-0.6b", "clip-vit-base-patch32", "youtu-embedding"],
            "description": "Embedding model to use.",
        },
        "backend": {
            "type": "string",
            "enum": ["torch", "sglang", "sglang-offline", "vllm", "vllm-http"],
            "default": "torch",
            "description": "Inference backend.",
        },
        "dataset": {
            "type": "string",
            "enum": ["yahoo_answers", "flickr8k", "synthetic_tokens", "synthetic_fixed_len"],
            "description": "Dataset to embed.",
        },
        "model_id": {"type": "string", "description": "Override model HF path or served name."},
        "dataset_path": {"type": "string", "description": "Path to dataset file (JSONL etc)."},
        "yahoo_mode": {"type": "string", "enum": ["q", "a", "q+a"], "default": "q"},
        "max_samples": {"type": "integer", "default": -1, "description": "Max samples (-1=all)."},
        "batch_size": {"type": "integer", "default": 128},
        "warmup_samples": {"type": "integer", "default": 1},
        "device": {"type": "string", "description": "e.g. cpu, cuda:0"},
        "base_url": {"type": "string", "description": "HTTP server URL (for sglang/vllm-http)."},
        "api": {"type": "string", "default": "v1"},
        "api_key": {"type": "string"},
        "timeout": {"type": "number", "default": 120.0, "description": "HTTP timeout (sec)."},
        "dtype": {"type": "string", "description": "bf16/fp16/fp32"},
        "normalize": {"type": "boolean", "default": True, "description": "L2 normalize embeddings."},
        "use_amx": {"type": "boolean", "default": False, "description": "Enable IPEX/AMX."},
        "profile": {"type": "boolean", "default": False},
        "synthetic_token_len": {"type": "integer", "description": "Token count for synthetic_tokens."},
        "synthetic_input_len": {"type": "integer", "description": "Char count for synthetic_fixed_len."},
        "max_length": {"type": "integer", "default": 512},
        "tp_size": {"type": "integer", "default": 1},
        "max_model_len": {"type": "integer", "default": 8192},
        "gpu_memory_utilization": {"type": "number", "default": 0.90},
        "encoding_format": {"type": "string"},
        "output_path": {"type": "string", "description": "Save embeddings tensor (.pt)."},
        "flickr8k_images_dir": {"type": "string"},
        "flickr8k_captions_file": {"type": "string"},
        "timeout_sec": {"type": "number", "default": 600, "description": "Skill execution timeout."},
    },
    "required": ["model", "dataset"],
}
