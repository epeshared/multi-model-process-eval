from __future__ import annotations

from typing import Any, Dict, List

from ..util import REPO_ROOT, run_cmd, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = str(args.get("out_dir") or "").strip()
    if not out_dir:
        raise ValueError("out_dir is required")

    sizes = str(args.get("sizes") or "224x224,384x384,512x512,1024x1024,1280x720,1920x1080").strip()
    per_size = int(args.get("per_size") or 4)
    pattern = str(args.get("pattern") or "checker").strip()
    fmt = str(args.get("format") or "png").strip()
    seed = int(args.get("seed") or 0)

    script = REPO_ROOT / "scripts/scale-test/vl-embedding/gen_test_images.py"
    argv: List[str] = [
        "python3", "-u", str(script),
        "--out", out_dir,
        "--sizes", sizes,
        "--per-size", str(per_size),
        "--pattern", pattern,
        "--format", fmt,
        "--seed", str(seed),
    ]

    rc, out = run_cmd(argv, cwd=REPO_ROOT)
    return {
        "returncode": rc,
        "command": argv,
        "output": truncate(out, limit=10000),
    }


SPEC = {
    "type": "object",
    "properties": {
        "out_dir": {
            "type": "string",
            "description": "Output directory for generated images.",
        },
        "sizes": {
            "type": "string",
            "default": "224x224,384x384,512x512,1024x1024,1280x720,1920x1080",
            "description": "Comma-separated WxH sizes (e.g. '224x224,512x512').",
        },
        "per_size": {
            "type": "integer",
            "default": 4,
            "description": "Number of images per resolution.",
        },
        "pattern": {
            "type": "string",
            "enum": ["checker", "gradient", "noise"],
            "default": "checker",
            "description": "Image content pattern.",
        },
        "format": {
            "type": "string",
            "enum": ["png", "jpg"],
            "default": "png",
        },
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "Deterministic seed for noise pattern.",
        },
    },
    "required": ["out_dir"],
}
