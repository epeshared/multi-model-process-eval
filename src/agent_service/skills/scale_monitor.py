from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..util import REPO_ROOT, resolve_repo_path, truncate


_MONITOR_SCRIPTS = {
    "embedding": "scripts/scale-test/embedding/monitor_scale_fix_token_len.sh",
    "vl-embedding": "scripts/scale-test/vl-embedding/monitor_scale_fix_image_size.sh",
}


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    scale_id = str(args.get("scale_id") or "").strip()
    if not scale_id:
        raise ValueError("scale_id is required")

    task = str(args.get("task") or "embedding").strip()
    result_root = str(args.get("result_root") or "").strip()

    monitor_rel = _MONITOR_SCRIPTS.get(task)
    if not monitor_rel:
        raise ValueError(f"No monitor script for task {task!r}. Available: {list(_MONITOR_SCRIPTS.keys())}")

    monitor = REPO_ROOT / monitor_rel
    if not monitor.exists():
        raise FileNotFoundError(f"Monitor script not found: {monitor}")

    # The monitor script loops forever (sleep $INTERVAL), so we set
    # --interval 1 and use a short timeout to capture one iteration.
    argv: List[str] = ["bash", str(monitor), "--scale-id", scale_id, "--interval", "1"]

    if result_root:
        argv += ["--result-root", result_root]

    try:
        p = subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=10,
        )
        rc = p.returncode
        out = p.stdout or ""
    except subprocess.TimeoutExpired as e:
        # Expected — we intentionally kill the infinite loop after one poll.
        rc = 0
        out = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else str(e.stdout or "")
    return {
        "returncode": rc,
        "task": task,
        "scale_id": scale_id,
        "command": argv,
        "output": truncate(out, limit=40000),
    }


SPEC = {
    "type": "object",
    "properties": {
        "scale_id": {
            "type": "string",
            "description": "The scale-test run ID to monitor.",
        },
        "task": {
            "type": "string",
            "enum": ["embedding", "vl-embedding"],
            "default": "embedding",
            "description": "Task type — determines which monitor script to invoke.",
        },
        "result_root": {
            "type": "string",
            "description": "Optional override for result root directory.",
        },
    },
    "required": ["scale_id"],
}
