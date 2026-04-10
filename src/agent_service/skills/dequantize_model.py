"""dequantize_model — convert FP8 weights to FP16/BF16 via dequantize_fp8_to_fp16.py."""
from __future__ import annotations

from typing import Any, Dict, List

from ..util import REPO_ROOT, run_cmd, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    in_model_dir = str(args.get("in_model_dir") or "").strip()
    out_model_dir = str(args.get("out_model_dir") or "").strip()
    if not in_model_dir:
        raise ValueError("in_model_dir is required")
    if not out_model_dir:
        raise ValueError("out_model_dir is required")

    argv: List[str] = [
        "python", "scripts/tools/dequantize_fp8_to_fp16.py",
        "--in-model-dir", in_model_dir,
        "--out-model-dir", out_model_dir,
    ]

    dtype = str(args.get("dtype") or "").strip()
    if dtype:
        argv.extend(["--dtype", dtype])
    if args.get("overwrite"):
        argv.append("--overwrite")
    if args.get("keep_quant_aux"):
        argv.append("--keep-quant-aux")
    if args.get("verbose"):
        argv.append("--verbose")

    timeout_s = float(args.get("timeout_sec") or 1800)
    rc, out = run_cmd(argv, cwd=REPO_ROOT, timeout_s=timeout_s)
    return {"returncode": rc, "command": argv, "output": truncate(out, limit=40000)}


SPEC = {
    "type": "object",
    "properties": {
        "in_model_dir": {"type": "string", "description": "Source FP8 model directory."},
        "out_model_dir": {"type": "string", "description": "Destination FP16/BF16 model directory."},
        "dtype": {
            "type": "string",
            "enum": ["float16", "bfloat16"],
            "default": "float16",
            "description": "Target dtype.",
        },
        "overwrite": {"type": "boolean", "default": False},
        "keep_quant_aux": {"type": "boolean", "default": False, "description": "Keep weight_scale/input_scale tensors."},
        "verbose": {"type": "boolean", "default": False},
        "timeout_sec": {"type": "number", "default": 1800},
    },
    "required": ["in_model_dir", "out_model_dir"],
}
