from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def truncate(s: str, limit: int = 20000) -> str:
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...<truncated>...\n"


def run_cmd(
    argv: List[str],
    *,
    cwd: Optional[Path] = None,
    timeout_s: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str]:
    p = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout_s,
        env=env,
    )
    return int(p.returncode), p.stdout or ""
