"""auto_test — orchestrate multi-config automated embedding tests via run_auto_test.py."""
from __future__ import annotations

from typing import Any, Dict, List

from ..util import REPO_ROOT, run_cmd, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    config_path = str(args.get("config_path") or "").strip()
    if not config_path:
        raise ValueError("config_path is required")

    argv: List[str] = [
        "python", "scripts/auto-test/embedding/run_auto_test.py",
        "--config", config_path,
    ]

    # Repeatable --only / --skip
    for name in (args.get("only") or []):
        argv.extend(["--only", str(name)])
    for name in (args.get("skip") or []):
        argv.extend(["--skip", str(name)])

    if args.get("dry_run"):
        argv.append("--dry-run")
    if args.get("tee"):
        argv.append("--tee")
    if args.get("restart_servers"):
        argv.append("--restart-servers")
    if args.get("stop_servers_after_job"):
        argv.append("--stop-servers-after-job")

    reparse = args.get("reparse_run_id")
    if reparse:
        argv.extend(["--reparse-run-id", str(reparse)])

    timeout_s = float(args.get("timeout_sec") or 7200)
    rc, out = run_cmd(argv, cwd=REPO_ROOT, timeout_s=timeout_s)
    return {"returncode": rc, "command": argv, "output": truncate(out, limit=40000)}


SPEC = {
    "type": "object",
    "properties": {
        "config_path": {
            "type": "string",
            "description": (
                "Path to auto-test config JSON "
                "(e.g. scripts/auto-test/embedding/config_fix_token_len.json)."
            ),
        },
        "only": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
            "description": "Run only jobs with these exact names.",
        },
        "skip": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
            "description": "Skip jobs with these exact names.",
        },
        "dry_run": {"type": "boolean", "default": False, "description": "Print commands without executing."},
        "tee": {"type": "boolean", "default": False, "description": "Stream logs to terminal and files."},
        "restart_servers": {"type": "boolean", "default": False, "description": "Kill and restart servers."},
        "stop_servers_after_job": {"type": "boolean", "default": False, "description": "Stop servers between jobs."},
        "reparse_run_id": {"type": "string", "description": "Re-parse old logs for a given run ID."},
        "timeout_sec": {"type": "number", "default": 7200, "description": "Timeout (auto-tests can be long)."},
    },
    "required": ["config_path"],
}
