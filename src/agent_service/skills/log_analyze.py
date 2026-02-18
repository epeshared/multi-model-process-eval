from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..util import resolve_repo_path


_PATTERNS: List[Tuple[str, re.Pattern[str], str]] = [
    (
        "ssh_proxy_timeout",
        re.compile(r"banner exchange|UNKNOWN port 65535|Connection timed out", re.IGNORECASE),
        "SSH/Proxy 连接超时（常见于 ProxyCommand/SOCKS/网络不通）",
    ),
    (
        "conda_missing",
        re.compile(r"conda: command not found", re.IGNORECASE),
        "远端 shell 找不到 conda（非交互 shell 未加载 conda.sh 或未安装 miniforge）",
    ),
    (
        "pip_failed",
        re.compile(r"pip.*(error|failed)|Could not find a version|ResolutionImpossible", re.IGNORECASE),
        "pip 安装依赖失败",
    ),
    (
        "permission",
        re.compile(r"Permission denied", re.IGNORECASE),
        "权限/密钥/用户名问题导致 SSH 或文件操作失败",
    ),
]


def _read(p: Path, max_bytes: int = 2_000_000) -> str:
    try:
        b = p.read_bytes()
        if len(b) > max_bytes:
            b = b[-max_bytes:]
        return b.decode("utf-8", errors="replace")
    except Exception:
        return ""


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    paths_in = args.get("paths")
    if not isinstance(paths_in, list) or not paths_in:
        raise ValueError("paths must be a non-empty list")

    findings: List[Dict[str, Any]] = []
    for p0 in paths_in:
        p = resolve_repo_path(str(p0))
        txt = _read(p)
        matched: List[Dict[str, str]] = []
        for key, pat, desc in _PATTERNS:
            if pat.search(txt):
                matched.append({"key": key, "description": desc})
        findings.append(
            {
                "path": str(p),
                "exists": p.exists(),
                "matches": matched,
            }
        )

    return {"findings": findings}


SPEC = {
    "type": "object",
    "properties": {
        "paths": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["paths"],
}
