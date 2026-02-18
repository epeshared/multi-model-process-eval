from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..util import REPO_ROOT, load_json, resolve_repo_path


def _resolve_result_root_from_config(cfg_path: Path) -> Path:
    obj = load_json(cfg_path)
    rr = str(obj.get("result_root") or "scripts/scale-test/embedding/result/fix_token_len").strip()
    return resolve_repo_path(rr)


def _read_aggregate_counts(csv_path: Path) -> Dict[str, Any]:
    total = 0
    ok = 0
    by_rc: Dict[str, int] = {}
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                rc = str(row.get("exit_code") or row.get("sweep_rc") or "").strip()
                if rc == "0":
                    ok += 1
                by_rc[rc] = by_rc.get(rc, 0) + 1
    except Exception:
        return {"total": 0, "ok": 0, "by_rc": {}}
    return {"total": total, "ok": ok, "by_rc": by_rc}


def _tail(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    config_path = str(args.get("config_path") or "").strip()
    scale_id = str(args.get("scale_id") or "").strip()

    if not scale_id:
        raise ValueError("scale_id is required")

    cfg: Optional[Path] = None
    result_root: Optional[Path] = None

    if config_path:
        cfg = resolve_repo_path(config_path)
        if not cfg.exists():
            raise FileNotFoundError(f"config not found: {cfg}")
        result_root = _resolve_result_root_from_config(cfg)

    if not result_root:
        rr_in = str(args.get("result_root") or "").strip()
        if rr_in:
            result_root = resolve_repo_path(rr_in)
        else:
            result_root = resolve_repo_path("scripts/scale-test/embedding/result/fix_token_len")

    run_dir = (result_root / scale_id).resolve()
    exists = run_dir.exists()

    out: Dict[str, Any] = {
        "scale_id": scale_id,
        "result_root": str(result_root),
        "run_dir": str(run_dir),
        "exists": exists,
    }
    if not exists:
        return out

    agg = run_dir / "aggregate.csv"
    out["aggregate"] = {
        "path": str(agg),
        "exists": agg.exists(),
        "counts": _read_aggregate_counts(agg) if agg.exists() else None,
    }

    # Multi-host layout: <run_dir>/hosts/<host>/...
    hosts_dir = run_dir / "hosts"
    hosts: List[Dict[str, Any]] = []
    if hosts_dir.exists():
        for host_dir in sorted([p for p in hosts_dir.iterdir() if p.is_dir()]):
            h: Dict[str, Any] = {"host": host_dir.name, "dir": str(host_dir)}
            h_agg = host_dir / "aggregate.csv"
            h["aggregate"] = {
                "exists": h_agg.exists(),
                "counts": _read_aggregate_counts(h_agg) if h_agg.exists() else None,
            }
            rlog = host_dir / "remote_run.log"
            h["remote_run_log"] = {"exists": rlog.exists(), "tail": _tail(rlog, n=50) if rlog.exists() else ""}
            hosts.append(h)
    out["hosts"] = hosts

    return out


SPEC = {
    "type": "object",
    "properties": {
        "config_path": {"type": "string", "description": "Optional config JSON; used to infer result_root."},
        "result_root": {"type": "string", "description": "Optional override result_root."},
        "scale_id": {"type": "string"},
    },
    "required": ["scale_id"],
}
