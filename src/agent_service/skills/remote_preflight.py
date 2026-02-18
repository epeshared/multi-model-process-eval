from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..util import REPO_ROOT, load_json, resolve_repo_path, truncate


def _ssh_argv(server: Dict[str, Any], cmd: str) -> List[str]:
    ip = str(server.get("ip") or "").strip()
    user = str(server.get("username") or "").strip() or "root"
    port = str(server.get("port") or "").strip()

    argv: List[str] = ["ssh", "-o", "BatchMode=yes"]
    if port:
        argv += ["-p", port]

    identity = str(server.get("identity_file") or "").strip()
    if identity:
        argv += ["-i", identity]

    for opt in (server.get("ssh_options") or []):
        opt_s = str(opt).strip()
        if opt_s:
            argv += ["-o", opt_s]

    target = f"{user}@{ip}" if user else ip
    argv += [target, "bash", "-lc", cmd]
    return argv


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    config_path = str(args.get("config_path") or "").strip()
    if not config_path:
        raise ValueError("config_path is required")

    cfg = resolve_repo_path(config_path)
    if not cfg.exists():
        raise FileNotFoundError(f"config not found: {cfg}")

    obj = load_json(cfg)
    run = obj.get("run") or {}
    servers = run.get("servers") or []
    if not isinstance(servers, list) or not servers:
        raise ValueError("config.run.servers is empty")

    remote_repo_dir = str(run.get("remote_repo_dir") or "").strip() or "/"
    remote_result_root = str(run.get("remote_result_root") or "").strip() or ""

    checks = (
        "set -e; "
        "echo '[preflight] whoami='$(whoami); "
        "echo '[preflight] hostname='$(hostname); "
        "if command -v conda >/dev/null 2>&1; then echo '[preflight] conda=PATH'; "
        "elif [ -x \"$HOME/miniforge3/bin/conda\" ]; then echo '[preflight] conda=$HOME/miniforge3/bin/conda'; "
        "else echo '[preflight] conda=MISSING'; fi; "
        f"if [ -d {shlex.quote(remote_repo_dir)} ]; then echo '[preflight] remote_repo_dir=OK'; else echo '[preflight] remote_repo_dir=MISSING'; fi; "
        + (f"if [ -d {shlex.quote(remote_result_root)} ]; then echo '[preflight] remote_result_root=OK'; else echo '[preflight] remote_result_root=MISSING'; fi; " if remote_result_root else "")
    )

    out_hosts: List[Dict[str, Any]] = []
    for s in servers:
        if not isinstance(s, dict):
            continue
        argv = _ssh_argv(s, checks)
        p = subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        out_hosts.append(
            {
                "ip": str(s.get("ip") or ""),
                "returncode": int(p.returncode),
                "command": argv,
                "output": truncate(p.stdout or "", limit=20000),
            }
        )

    return {
        "config_path": str(cfg),
        "remote_repo_dir": remote_repo_dir,
        "remote_result_root": remote_result_root,
        "hosts": out_hosts,
    }


SPEC = {
    "type": "object",
    "properties": {
        "config_path": {"type": "string"}
    },
    "required": ["config_path"],
}
