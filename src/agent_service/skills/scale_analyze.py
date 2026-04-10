from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..util import REPO_ROOT, resolve_repo_path, run_cmd, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    run_dir_in = str(args.get("run_dir") or "").strip()
    if not run_dir_in:
        raise ValueError("run_dir is required")

    run_dir = resolve_repo_path(run_dir_in)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    agg = run_dir / "aggregate.csv"
    if not agg.exists():
        raise FileNotFoundError(f"aggregate.csv not found in {run_dir}")

    out_dir = str(args.get("out_dir") or "analysis").strip()

    socket_metrics: List[str] = args.get("socket_metrics") or []

    analyzer = REPO_ROOT / "scripts/scale-test/embedding/analyze_run.py"
    argv: List[str] = ["python3", "-u", str(analyzer), str(run_dir), "--out-dir", out_dir]

    if socket_metrics:
        argv += ["--socket-metrics"] + [str(m) for m in socket_metrics]

    rc, out = run_cmd(argv, cwd=REPO_ROOT)

    analysis_dir = run_dir / out_dir
    generated: List[str] = []
    if analysis_dir.exists():
        generated = sorted(str(p.relative_to(run_dir)) for p in analysis_dir.rglob("*") if p.is_file())

    return {
        "returncode": rc,
        "command": argv,
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "generated_files": generated,
        "output": truncate(out, limit=40000),
    }


SPEC = {
    "type": "object",
    "properties": {
        "run_dir": {
            "type": "string",
            "description": "Path to scale-test run directory containing aggregate.csv (repo-relative or absolute).",
        },
        "out_dir": {
            "type": "string",
            "default": "analysis",
            "description": "Output subdirectory name under run_dir (default: 'analysis').",
        },
        "socket_metrics": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
            "description": "Optional EMON socket-view metric keys to extract.",
        },
    },
    "required": ["run_dir"],
}
