from __future__ import annotations

import signal
import subprocess
from typing import Any, Dict, Optional

from ..util import REPO_ROOT, truncate


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    action = str(args.get("action") or "start").strip().lower()
    port = int(args.get("port") or 8080)
    host = str(args.get("host") or "0.0.0.0").strip()

    server_py = REPO_ROOT / "scripts/scale-test/web/server.py"
    scale_test_root = REPO_ROOT / "scripts/scale-test"

    if action == "status":
        # Check if anything is listening on the port
        rc, out = _check_port(port)
        return {"action": "status", "port": port, "listening": rc == 0, "output": out}

    if action == "stop":
        rc, out = _stop_port(port)
        return {"action": "stop", "port": port, "returncode": rc, "output": out}

    if action == "start":
        # Check if already running
        rc, _ = _check_port(port)
        if rc == 0:
            return {"action": "start", "port": port, "already_running": True, "url": f"http://{host}:{port}/"}

        argv = [
            "python3", "-u", str(server_py),
            "--host", host,
            "--port", str(port),
            "--scale-test-root", str(scale_test_root),
        ]
        proc = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        # Wait briefly to see if it starts OK
        try:
            out = proc.stdout.read(2048) if proc.stdout else ""  # type: ignore[union-attr]
        except Exception:
            out = ""

        return {
            "action": "start",
            "port": port,
            "pid": proc.pid,
            "url": f"http://{host}:{port}/",
            "command": argv,
            "startup_output": truncate(out, limit=2000),
        }

    raise ValueError(f"Unknown action: {action!r}. Must be 'start', 'stop', or 'status'.")


def _check_port(port: int) -> tuple:
    """Check if port is in use via ss/lsof."""
    try:
        p = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        lines = [l for l in p.stdout.strip().splitlines() if str(port) in l]
        if lines:
            return 0, "\n".join(lines)
        return 1, ""
    except Exception as e:
        return 1, str(e)


def _stop_port(port: int) -> tuple:
    """Kill process listening on port."""
    try:
        p = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, str(e)


SPEC = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["start", "stop", "status"],
            "default": "start",
            "description": "Server action: start, stop, or check status.",
        },
        "port": {
            "type": "integer",
            "default": 8080,
            "description": "HTTP port for the web server.",
        },
        "host": {
            "type": "string",
            "default": "0.0.0.0",
            "description": "Bind address.",
        },
    },
    "required": [],
}
