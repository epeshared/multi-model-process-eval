from __future__ import annotations

from typing import Any, Dict, List

from ..util import REPO_ROOT, resolve_repo_path, run_cmd, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    config_path = str(args.get("config_path") or "").strip()
    if not config_path:
        raise ValueError("config_path is required")

    cfg = resolve_repo_path(config_path)
    if not cfg.exists():
        raise FileNotFoundError(f"config not found: {cfg}")

    scale_id = str(args.get("scale_id") or "").strip()
    resume = bool(args.get("resume") or False)
    tee = bool(args.get("tee") or False)
    dry_run = bool(args.get("dry_run") or False)

    extra_args_in = args.get("extra_args") or []
    if not isinstance(extra_args_in, list):
        raise ValueError("extra_args must be a list")
    extra_args: List[str] = [str(x) for x in extra_args_in]

    runner = REPO_ROOT / "scripts/scale-test/embedding/run_scale_fix_token_len.py"
    argv: List[str] = [
        "python3",
        "-u",
        str(runner),
        "--config",
        str(cfg),
    ]
    if scale_id:
        argv += ["--scale-id", scale_id]
    if resume:
        argv += ["--resume"]
    if tee:
        argv += ["--tee"]
    if dry_run:
        argv += ["--dry-run"]
    argv += extra_args

    rc, out = run_cmd(argv, cwd=REPO_ROOT)
    return {
        "returncode": rc,
        "command": argv,
        "output": truncate(out, limit=40000),
    }


SPEC = {
    "type": "object",
    "properties": {
        "config_path": {"type": "string", "description": "Path to scale-test config JSON (repo-relative or absolute)."},
        "scale_id": {"type": "string", "description": "Optional fixed scale_id for <result_root>/<scale_id>/."},
        "resume": {"type": "boolean", "default": False},
        "tee": {"type": "boolean", "default": False},
        "dry_run": {"type": "boolean", "default": False},
        "extra_args": {"type": "array", "items": {"type": "string"}, "default": []},
    },
    "required": ["config_path"],
}
