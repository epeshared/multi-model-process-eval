#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import mimetypes
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse


WEB_DIR = Path(__file__).resolve().parent
SCALE_TEST_ROOT_DEFAULT = (WEB_DIR / "..").resolve()


@dataclass(frozen=True)
class RunRef:
    task: str
    suite: str
    run_id: str


@dataclass(frozen=True)
class RunInfo:
    ref: RunRef
    run_dir: Path
    analysis_dir: Path
    mtime: float


# Best-effort metadata cache: {run_dir: (mtime, meta_dict)}
_RUN_META_CACHE: Dict[str, Tuple[float, Dict[str, str]]] = {}


def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _fmt_dt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _safe_relpath(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()


def _sanitize_download_name(name: str, *, default_ext: Optional[str] = None, max_len: int = 200) -> str:
    """Sanitize a user-provided download filename.

    - Removes path separators
    - Keeps alnum, '.', '-', '_' and replaces others with '_'
    - Optionally enforces an extension
    """
    s = (name or "").strip()
    # drop any path component
    s = s.split("/")[-1].split("\\")[-1]
    if not s:
        s = "download"
    out = []
    for ch in s:
        if ch.isalnum() or ch in {".", "-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    s2 = "".join(out)
    while "__" in s2:
        s2 = s2.replace("__", "_")
    s2 = s2.strip("._-") or "download"
    if default_ext:
        ext = default_ext if default_ext.startswith(".") else f".{default_ext}"
        if not s2.lower().endswith(ext.lower()):
            s2 = s2 + ext
    if len(s2) > max_len:
        # keep extension if present
        suf = ""
        if "." in s2:
            base, dot, ext = s2.rpartition(".")
            if base and ext:
                suf = dot + ext
                s2 = base
        s2 = s2[: max(1, max_len - len(suf))] + suf
    return s2


def discover_runs(scale_test_root: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    for task_dir in sorted(scale_test_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        if task in {"web", "__pycache__"}:
            continue
        result_dir = task_dir / "result"
        if not result_dir.exists() or not result_dir.is_dir():
            continue

        for suite_dir in sorted(result_dir.iterdir()):
            if not suite_dir.is_dir():
                continue
            suite = suite_dir.name

            for run_dir in sorted(suite_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name
                analysis_dir = run_dir / "analysis"
                if not analysis_dir.exists() or not analysis_dir.is_dir():
                    continue

                try:
                    mtime = run_dir.stat().st_mtime
                except Exception:
                    mtime = 0.0

                runs.append(
                    RunInfo(
                        ref=RunRef(task=task, suite=suite, run_id=run_id),
                        run_dir=run_dir,
                        analysis_dir=analysis_dir,
                        mtime=mtime,
                    )
                )

    # Newest first
    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


def _read_text(path: Path, max_bytes: int = 256_000) -> str:
    data = path.read_bytes()[:max_bytes]
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("utf-8", errors="replace")


def _list_analysis_files(analysis_dir: Path) -> Tuple[List[Path], List[Path]]:
    pngs: List[Path] = []
    csvs: List[Path] = []
    for p in sorted(analysis_dir.iterdir()):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".png":
            pngs.append(p)
        elif suf == ".csv":
            csvs.append(p)
    return pngs, csvs


def _extract_run_meta(run: RunInfo) -> Dict[str, str]:
    """Best-effort extract run metadata (currently: model) for display/filter.

    Sources (in priority order):
    - model: run_dir/aggregate.csv (column: model or model_id)
    - model fallback: */auto_test_config.generated.json (jobs[0].env.MODEL / MODEL_ID)
    """
    key = str(run.run_dir)
    cached = _RUN_META_CACHE.get(key)
    if cached and cached[0] == run.mtime:
        return cached[1]

    meta: Dict[str, str] = {"model": ""}

    # 1) model from aggregate.csv
    try:
        agg = run.run_dir / "aggregate.csv"
        if agg.exists():
            with agg.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for i, r in enumerate(reader):
                    if i > 300:
                        break
                    m = (r.get("model") or "").strip()
                    mid = (r.get("model_id") or "").strip()
                    if m:
                        meta["model"] = m
                        break
                    if mid and not meta["model"]:
                        meta["model"] = mid
                # If still empty, fall back to header-only rows.
    except Exception:
        pass

    # 2) model fallback from auto_test_config.generated.json (first job env)
    def _try_model_from(p: Path) -> None:
        if meta.get("model"):
            return
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            jobs = obj.get("jobs") or []
            if not jobs:
                return
            env = (jobs[0].get("env") or {}) if isinstance(jobs[0], dict) else {}
            model2 = str(env.get("MODEL") or "").strip()
            model_id2 = str(env.get("MODEL_ID") or "").strip()
            if not meta.get("model"):
                if model2:
                    meta["model"] = model2
                elif model_id2:
                    meta["model"] = model_id2
        except Exception:
            return

    try:
        # Prefer variants.json order if present.
        v = run.run_dir / "variants.json"
        if v.exists():
            obj = json.loads(v.read_text(encoding="utf-8"))
            variants = obj.get("variants") or []
            if variants and isinstance(variants, list):
                first = variants[0]
                d = first.get("dir") if isinstance(first, dict) else ""
                if d:
                    p = Path(str(d)) / "auto_test_config.generated.json"
                    if p.exists():
                        _try_model_from(p)
        if not meta.get("model"):
            # Fallback: find any generated config under the run.
            for p in sorted(run.run_dir.glob("**/auto_test_config.generated.json")):
                _try_model_from(p)
                if meta.get("model"):
                    break
    except Exception:
        pass

    _RUN_META_CACHE[key] = (run.mtime, meta)
    return meta


def _maybe_autogen_analysis(*, scale_test_root: Path, run: RunInfo, required: List[Path]) -> Tuple[bool, str]:
    """Ensure required analysis artifacts exist.

    If missing, try to run task-specific analyze_run.py:
      scripts/scale-test/<task>/analyze_run.py <run_dir>

    Returns (ok, message). Always best-effort; never raises.
    """
    # If everything exists, skip.
    if all(p.exists() for p in required):
        return True, ""

    autogen = os.environ.get("SCALE_TEST_WEB_AUTOGEN", "1").strip().lower() not in {"0", "false", "no"}
    if not autogen:
        return False, "autogen disabled (set SCALE_TEST_WEB_AUTOGEN=1 to enable)"

    analyzer = (scale_test_root / run.ref.task / "analyze_run.py").resolve()
    if not analyzer.exists():
        return False, f"analyzer not found: {analyzer}"

    lock = run.analysis_dir / ".autogen.lock"
    # Acquire lock (best-effort). If another thread/process is generating, wait briefly.
    lock_fd: Optional[int] = None
    for _ in range(30):
        try:
            lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            time.sleep(0.25)
            if all(p.exists() for p in required):
                return True, ""
        except Exception as e:
            return False, f"lock error: {e}"

    if lock_fd is None:
        # Timed out waiting for another generator.
        return all(p.exists() for p in required), "autogen busy (lock held)"

    try:
        # Ensure analysis dir exists.
        run.analysis_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(analyzer), str(run.run_dir)]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        ok = proc.returncode == 0 and all(p.exists() for p in required)
        msg = "" if ok else (proc.stdout[-4000:] if proc.stdout else f"analyze rc={proc.returncode}")
        # Save last autogen log for debugging.
        try:
            (run.analysis_dir / "web_autogen.log").write_text(proc.stdout or "", encoding="utf-8")
        except Exception:
            pass
        return ok, msg
    except subprocess.TimeoutExpired:
        return False, "analyze_run.py timed out"
    except Exception as e:
        return False, f"autogen error: {e}"
    finally:
        try:
            os.close(lock_fd)
        except Exception:
            pass
        try:
            lock.unlink()
        except Exception:
            pass


def _csv_numeric_summary(path: Path, *, max_rows: int = 200_000) -> Dict[str, Any]:
    """Dependency-free numeric summary for a CSV.

    Returns:
      {
        rows: int,
        cols: int,
        numeric: {col: {count, mean, min, max}}
      }
    """
    text = _read_text(path, max_bytes=20_000_000)
    buf = io.StringIO(text)
    reader = csv.reader(buf)
    header = next(reader, None)
    if not header:
        return {"rows": 0, "cols": 0, "numeric": {}}

    ncols = len(header)
    # Welford-ish mean + min/max
    count: Dict[int, int] = {}
    mean: Dict[int, float] = {}
    vmin: Dict[int, float] = {}
    vmax: Dict[int, float] = {}
    rows = 0

    for row in reader:
        rows += 1
        if rows > max_rows:
            break
        # pad
        if len(row) < ncols:
            row = row + [""] * (ncols - len(row))
        for i, cell in enumerate(row[:ncols]):
            s = str(cell).strip()
            if not s:
                continue
            try:
                x = float(s)
            except Exception:
                continue
            c = count.get(i, 0) + 1
            count[i] = c
            if c == 1:
                mean[i] = x
                vmin[i] = x
                vmax[i] = x
            else:
                mean[i] = mean[i] + (x - mean[i]) / c
                if x < vmin[i]:
                    vmin[i] = x
                if x > vmax[i]:
                    vmax[i] = x

    numeric: Dict[str, Any] = {}
    for i, c in count.items():
        col = header[i]
        numeric[col] = {
            "count": c,
            "mean": mean.get(i),
            "min": vmin.get(i),
            "max": vmax.get(i),
        }
    return {"rows": rows, "cols": ncols, "numeric": numeric}


def _safe_float(s: Any) -> Optional[float]:
    try:
        if s is None:
            return None
        st = str(s).strip()
        if not st or st.lower() in {"nan", "none"}:
            return None
        return float(st)
    except Exception:
        return None


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _quantile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs)
    deny = sum((y - my) ** 2 for y in ys)
    if denx <= 0 or deny <= 0:
        return None
    return num / (denx**0.5 * deny**0.5)


def _pill(label: str, level: str) -> str:
    # level: ok|warn|strong
    return f'<span class="pill {level}">{_e(label)}</span>'


def _impact_from_ratio(r: Optional[float]) -> Tuple[str, str]:
    if r is None:
        return ("", "")
    if r >= 3.0:
        return ("高", "strong")
    if r >= 1.5:
        return ("中", "warn")
    return ("低", "ok")


def _impact_from_eff(e: Optional[float]) -> Tuple[str, str]:
    if e is None:
        return ("", "")
    if e >= 0.80:
        return ("好", "ok")
    if e >= 0.60:
        return ("一般", "warn")
    return ("较差", "strong")


def _render_scaling_csv_ppt_summary(csv_path: Path) -> str:
    """PPT-ready one-page summary for *_scaling.csv (stdlib-only).

    Returns an HTML fragment.
    """
    name = csv_path.name
    text = _read_text(csv_path, max_bytes=8_000_000)
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows or not reader.fieldnames:
        return '<div class="muted">Empty CSV.</div>'

    cols = set(reader.fieldnames)

    def _norm_key(v: Any) -> str:
        fv = _safe_float(v)
        if fv is None:
            return str(v).strip()
        if abs(fv - int(fv)) < 1e-9:
            return str(int(fv))
        return f"{fv:g}"

    # Determine schema per CSV.
    if name == "token_len_scaling.csv":
        x = "token_len"
        group_cols = ["cpu_count", "kv_cap", "batch_size"]
        y = "tokens_per_sec" if "tokens_per_sec" in cols else "tps"
        corr_y = "tokens_per_sec" if "tokens_per_sec" in cols else None
        takeaway = "token_len 改变会显著影响单位时间可处理 token 数；通常 token_len 越大 tokens/sec 越低。"
    elif name == "batch_size_scaling.csv":
        x = "batch_size"
        group_cols = ["cpu_count", "kv_cap", "token_len"]
        y = "tps"
        corr_y = None
        takeaway = "batch_size 主要影响吞吐上限；总体影响常小于 token_len / kv_cap。"
    elif name == "kv_scaling.csv":
        x = "kv_cap"
        group_cols = ["cpu_count", "batch_size", "token_len"]
        y = "tps"
        corr_y = None
        takeaway = "KV cap 往往是关键瓶颈维度；kv_cap 不足会显著拉低 TPS。"
    elif name == "cpu_scaling.csv":
        # special: speedup/efficiency
        need = {"cpu_count", "tps", "kv_cap", "batch_size", "token_len"}
        if not need.issubset(cols):
            missing = ", ".join(sorted(need - cols))
            return f'<div class="muted">Missing required columns: {_e(missing)}</div>'
        groups: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}
        for r in rows:
            cpu = _safe_float(r.get("cpu_count"))
            tps = _safe_float(r.get("tps"))
            if cpu is None or tps is None:
                continue
            key = (_norm_key(r.get("kv_cap")), _norm_key(r.get("batch_size")), _norm_key(r.get("token_len")))
            groups.setdefault(key, []).append((cpu, tps))

        speedups: List[float] = []
        effs: List[float] = []
        for pts in groups.values():
            pts2 = sorted(pts, key=lambda t: t[0])
            if len({p[0] for p in pts2}) < 2:
                continue
            min_cpu, tps_min = pts2[0]
            max_cpu, tps_max = pts2[-1]
            if min_cpu <= 0 or max_cpu <= 0 or tps_min <= 0:
                continue
            speedup = tps_max / tps_min
            ideal = max_cpu / min_cpu
            eff = speedup / ideal
            speedups.append(speedup)
            effs.append(eff)

        med_speed = _median(speedups)
        med_eff = _median(effs)
        lvl, cls = _impact_from_eff(med_eff)
        rec = ""
        if med_eff is not None:
            if med_eff >= 0.80:
                rec = "CPU 扩展性较好：加核能有效提吞吐；注意 NUMA/绑核一致性。"
            elif med_eff >= 0.60:
                rec = "CPU 扩展性一般：优先排查线程/NUMA/带宽瓶颈，再决定是否加核。"
            else:
                rec = "CPU 扩展性较差：单纯加核收益有限，优先解决瓶颈（kv_cap/内存/调度）。"

        items = [
            f"<li><b>一句话结论</b>：CPU 核数增加能提升吞吐，但总体呈次线性；需要关注并行效率。</li>",
        ]
        if lvl:
            items.append(f"<li><b>扩展性评价</b>：{_pill(lvl, cls)}（以效率衡量）</li>")
        items.append(f"<li><b>统计口径</b>：rows={len(rows)}，groups={len(groups)}（按 kv_cap,batch_size,token_len 分组）</li>")
        if med_speed is not None:
            items.append(f"<li><b>中位加速比</b>：min→max cpu_count 的 median speedup=<b>{med_speed:.2f}×</b></li>")
        if med_eff is not None:
            items.append(f"<li><b>中位并行效率</b>：median efficiency vs ideal linear=<b>{med_eff:.2f}</b></li>")
        if rec:
            items.append(f"<li><b>建议</b>：{_e(rec)}</li>")
        return f"<ul>{''.join(items)}</ul>"
    else:
        return '<div class="muted">Not a scaling CSV.</div>'

    # Generic (token/batch/kv) summary with grouping.
    need = {x, y, *group_cols}
    if not need.issubset(cols):
        missing = ", ".join(sorted(need - cols))
        return f'<div class="muted">Missing required columns: {_e(missing)}</div>'

    groups2: Dict[Tuple[str, ...], List[Tuple[float, float, Optional[float]]]] = {}
    for r in rows:
        xv = _safe_float(r.get(x))
        yv = _safe_float(r.get(y))
        if xv is None or yv is None:
            continue
        cy: Optional[float] = None
        if corr_y is not None:
            cy = _safe_float(r.get(corr_y))
        key = tuple(_norm_key(r.get(c)) for c in group_cols)
        groups2.setdefault(key, []).append((xv, yv, cy))

    best_xs: List[float] = []
    ratios: List[float] = []
    cors: List[float] = []
    for pts in groups2.values():
        # require >=2 distinct x
        if len({p[0] for p in pts}) < 2:
            continue
        # best-x by y
        px = max(pts, key=lambda t: t[1])[0]
        best_xs.append(px)
        ys = [p[1] for p in pts]
        ymin = min(ys)
        ymax = max(ys)
        if ymin > 0:
            ratios.append(ymax / ymin)
        if corr_y is not None:
            xs = [p[0] for p in pts if p[2] is not None]
            ys2 = [p[2] for p in pts if p[2] is not None]
            if len(xs) >= 2 and len(set(xs)) >= 2:
                cc = _pearson_corr(xs, ys2)
                if cc is not None:
                    cors.append(cc)

    med_best = _median(best_xs)
    med_ratio = _median(ratios)
    p90_ratio = _quantile(ratios, 0.90)
    med_corr = _median(cors)

    lvl, cls = _impact_from_ratio(med_ratio)
    rec = ""
    if name == "kv_scaling.csv" and med_ratio is not None and med_ratio >= 1.5:
        rec = "KV cap 影响较大：优先确认不会被 KV 上限卡住（必要时提升 kv_cap / SGLANG_MAX_TOTAL_TOKENS）。"
    elif name == "batch_size_scaling.csv" and med_ratio is not None and med_ratio < 1.2:
        rec = "batch_size 总体影响较小：可优先以延迟与内存压力为约束来选 batch_size。"
    elif name == "token_len_scaling.csv" and med_corr is not None and med_corr <= -0.3:
        rec = "token_len 增大时 tokens/sec 往往下降：建议按目标场景 token_len 分层对比，避免用单一 token_len 代表全部。"
    elif med_best is not None:
        rec = f"建议优先选择 {x}≈{int(round(med_best))}（按当前 sweep 的中位最优点）。"

    items2: List[str] = []
    items2.append(f"<li><b>一句话结论</b>：{_e(takeaway)}</li>")
    if lvl:
        items2.append(f"<li><b>总体影响等级</b>：{_pill(lvl, cls)}（以组内 max/min 衡量影响强度）</li>")
    items2.append(f"<li><b>统计口径</b>：rows={len(rows)}，groups={len(groups2)}（按 {', '.join(group_cols)} 分组）</li>")
    if med_best is not None:
        items2.append(f"<li><b>中位最优点</b>：{_e(x)}≈<b>{med_best:.0f}</b>（argmax {_e(y)}）</li>")
    if med_ratio is not None:
        p90_s = f"{p90_ratio:.2f}×" if p90_ratio is not None else ""
        items2.append(f"<li><b>影响幅度</b>：组内 {_e(y)} 的 median(max/min)=<b>{med_ratio:.2f}×</b>" + (f"（p90 {p90_s}）" if p90_s else "") + "</li>")
    if corr_y is not None and med_corr is not None:
        items2.append(f"<li><b>趋势</b>：median corr({_e(x)}, {_e(corr_y)})=<b>{med_corr:.2f}</b></li>")
    if rec:
        items2.append(f"<li><b>建议</b>：{_e(rec)}</li>")
    return f"<ul>{''.join(items2)}</ul>"


def _parse_cpu_expr(cpu_expr: str) -> Optional[int]:
    """Best-effort parse of CPU expr like '0-15' or '0,2,4' into count."""
    s = str(cpu_expr).strip()
    if not s:
        return None
    # Prefer simple N-M ranges.
    if "," not in s and "-" in s:
        try:
            a, b = s.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i >= a_i:
                return b_i - a_i + 1
        except Exception:
            pass
    # Fallback: count comma separated.
    try:
        parts = [x.strip() for x in s.split(",") if x.strip()]
        if parts:
            return len(parts)
    except Exception:
        pass
    return None


def _html_page(title: str, body: str) -> str:
    css_href = "/static/style.css"
    return (
        "<!doctype html>\n"
        "<html>\n"
        "  <head>\n"
        '    <meta charset="utf-8" />\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"    <title>{_e(title)}</title>\n"
        f'    <link rel="stylesheet" href="{css_href}" />\n'
        "  </head>\n"
        "  <body>\n"
        "    <header class=\"topbar\">\n"
        "      <div class=\"container\">\n"
        "        <div class=\"brand\"><a href=\"/\">Scale Test Results</a></div>\n"
        "        <div class=\"hint\">auto-discovered from scripts/scale-test/*/result/**/analysis</div>\n"
        "      </div>\n"
        "    </header>\n"
        "    <main class=\"container\">\n"
        f"{body}\n"
        "    </main>\n"
        "    <footer class=\"footer\">\n"
        "      <div class=\"container\">Generated by scripts/scale-test/web/server.py</div>\n"
        "    </footer>\n"
        "  </body>\n"
        "</html>\n"
    )


def _render_home(scale_test_root: Path, runs: List[RunInfo], q: Dict[str, List[str]]) -> str:
    task_filter = (q.get("task") or [""])[0].strip()
    suite_filter = (q.get("suite") or [""])[0].strip()
    model_filter = (q.get("model") or [""])[0].strip()
    limit_s = (q.get("limit") or ["50"])[0].strip()
    try:
        limit = max(1, min(500, int(limit_s)))
    except Exception:
        limit = 50

    filtered = []
    for r in runs:
        if task_filter and r.ref.task != task_filter:
            continue
        if suite_filter and r.ref.suite != suite_filter:
            continue
        if model_filter:
            meta = _extract_run_meta(r)
            m = (meta.get("model") or "").lower()
            if model_filter.lower() not in m:
                continue
        filtered.append(r)

    # Task summary
    by_task: Dict[str, int] = {}
    for r in runs:
        by_task[r.ref.task] = by_task.get(r.ref.task, 0) + 1

    tasks_html = "".join(
        f'<a class="pill" href="/?task={_e(t)}">{_e(t)} <span class="pill-count">{n}</span></a>'
        for t, n in sorted(by_task.items())
    )

    rows = []
    for r in filtered[:limit]:
        meta = _extract_run_meta(r)
        model = meta.get("model") or "-"
        href = f"/run/{_e(r.ref.task)}/{_e(r.ref.suite)}/{_e(r.ref.run_id)}"
        rows.append(
            "<tr>"
            f"<td><a href=\"{href}\">{_e(r.ref.task)}</a></td>"
            f"<td>{_e(r.ref.suite)}</td>"
            f"<td><a href=\"{href}\">{_e(r.ref.run_id)}</a></td>"
            f"<td class=\"mono\">{_e(model)}</td>"
            f"<td class=\"mono\">{_e(_fmt_dt(r.mtime))}</td>"
            f"<td class=\"mono\">{_e(_safe_relpath(r.analysis_dir, scale_test_root))}</td>"
            "</tr>"
        )

    body = f"""
<section class="card">
  <h1>Scale Test Results</h1>
  <div class="sub">Root: <span class="mono">{_e(scale_test_root)}</span></div>

  <div class="toolbar">
    <form method="get" action="/" class="form-inline">
      <label>task <input name="task" value="{_e(task_filter)}" placeholder="embedding / vl / omni" /></label>
      <label>suite <input name="suite" value="{_e(suite_filter)}" placeholder="fix_token_len" /></label>
            <label>model <input name="model" value="{_e(model_filter)}" placeholder="qwen3-embedding-4b / Qwen3-Embedding-4B" /></label>
      <label>limit <input name="limit" value="{_e(limit)}" size="4" /></label>
      <button type="submit">Filter</button>
      <a class="btn" href="/">Reset</a>
    </form>
  </div>

  <div class="pills">{tasks_html}</div>
</section>

<section class="card">
  <h2>Recent Runs</h2>
  <table class="table">
    <thead>
      <tr>
        <th>task</th>
        <th>suite</th>
        <th>run</th>
                <th>model</th>
        <th>mtime</th>
        <th>analysis dir</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows) if rows else '<tr><td colspan="5" class="muted">No runs found.</td></tr>'}
    </tbody>
  </table>
</section>
"""
    return _html_page("Scale Test Results", body)


def _render_run(scale_test_root: Path, run: RunInfo, q: Dict[str, List[str]]) -> str:
    _, csvs = _list_analysis_files(run.analysis_dir)

    # Auto-generate summary if missing.
    summary_path = (run.analysis_dir / "run_summary.html").resolve()
    ok, msg = _maybe_autogen_analysis(scale_test_root=scale_test_root, run=run, required=[summary_path])

    # Run-level summary (pre-generated by analyze_run.py)
    summary_html = ""
    if _is_within(summary_path, run.analysis_dir) and summary_path.exists():
        # Embed as-is; content is generated by our analysis script.
        summary_html = _read_text(summary_path, max_bytes=2_000_000)
    else:
        hint = '<div class="muted">No run_summary.html found. Run: analyze_run.py</div>'
        if msg:
            hint += f'<div class="warn mono">autogen: {_e(msg)}</div>'
        summary_html = hint

    # CSV list
    csv_rows = []
    for c in csvs:
        rel = c.relative_to(run.run_dir).as_posix()
        raw = f"/raw/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/{_e(rel)}"
        preview = f"/csv/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/{_e(c.name)}"
        csv_rows.append(
            "<tr>"
            f"<td class=\"mono\">{_e(c.name)}</td>"
            f"<td><a href=\"{preview}\">Preview</a></td>"
            f"<td><a href=\"{raw}\" target=\"_blank\">Download</a></td>"
            "</tr>"
        )

    title = f"{run.ref.task} / {run.ref.suite} / {run.ref.run_id}"
    body = f"""
<section class="card">
  <div class="breadcrumbs">
    <a href="/">Home</a>
    <span class="sep">/</span>
    <span class="mono">{_e(run.ref.task)}</span>
    <span class="sep">/</span>
    <span class="mono">{_e(run.ref.suite)}</span>
    <span class="sep">/</span>
    <span class="mono">{_e(run.ref.run_id)}</span>
  </div>

  <h1>{_e(title)}</h1>
  <div class="sub">mtime: <span class="mono">{_e(_fmt_dt(run.mtime))}</span></div>
  <div class="sub">run_dir: <span class="mono">{_e(run.run_dir)}</span></div>
  <div class="sub">analysis_dir: <span class="mono">{_e(run.analysis_dir)}</span></div>
</section>

<section class="card">
    <h2>Summary</h2>
    <div class="embed">{summary_html}</div>
</section>

<section class="card">
  <h2>CSVs</h2>
  <table class="table">
    <thead><tr><th>file</th><th>preview</th><th>download</th></tr></thead>
    <tbody>
      {''.join(csv_rows) if csv_rows else '<tr><td colspan="3" class="muted">No CSVs found.</td></tr>'}
    </tbody>
  </table>
</section>
"""
    return _html_page(title, body)


def _render_csv_detail(scale_test_root: Path, run: RunInfo, csv_name: str, q: Dict[str, List[str]]) -> str:
    csv_path = (run.analysis_dir / csv_name).resolve()
    if not _is_within(csv_path, run.analysis_dir) or not csv_path.exists() or csv_path.suffix.lower() != ".csv":
        return _html_page("CSV not found", f"<section class=\"card\"><h1>CSV not found</h1><div class=\"mono\">{_e(csv_name)}</div></section>")

    # If plots/summaries are missing for this CSV, try auto-generate.
    required: List[Path] = [run.analysis_dir / "run_summary.html"]
    if csv_name == "token_len_scaling.csv":
        required += [run.analysis_dir / "plot_token_len_scaling.png"]
    elif csv_name == "batch_size_scaling.csv":
        required += [run.analysis_dir / "plot_batch_size_scaling.png"]
    elif csv_name == "cpu_scaling.csv":
        required += [run.analysis_dir / "plot_cpu_scaling.png"]
    elif csv_name == "kv_scaling.csv":
        required += [run.analysis_dir / "plot_kv_cap_scaling.png"]
    elif csv_name == "emon_socket_metrics.csv":
        # Per-job pies are driven by a manifest produced by analyze_run.py
        required += [run.analysis_dir / "emon_job_pies_manifest.json"]
    _maybe_autogen_analysis(scale_test_root=scale_test_root, run=run, required=required)

    # Special-case: emon_socket_metrics.csv page should only show preview.
    preview_only = csv_name == "emon_socket_metrics.csv"

    # Summary stats (skip for preview-only pages)
    num_rows = 0
    num_cols = 0
    num_table = ""
    if not preview_only:
        summ = _csv_numeric_summary(csv_path)
        numeric = summ.get("numeric") or {}
        num_rows = int(summ.get("rows") or 0)
        num_cols = int(summ.get("cols") or 0)

        # show up to N numeric columns
        num_items = list(numeric.items())
        num_items.sort(key=lambda kv: -(kv[1].get("count") or 0))
        num_items = num_items[:40]

    def fmtf(x: Any) -> str:
        try:
            return f"{float(x):.4g}"
        except Exception:
            return ""

        num_table_rows = []
        for col, st in num_items:
            num_table_rows.append(
                "<tr>"
                f"<td class=\"mono\">{_e(col)}</td>"
                f"<td class=\"mono\">{_e(st.get('count',''))}</td>"
                f"<td class=\"mono\">{_e(fmtf(st.get('mean')))}</td>"
                f"<td class=\"mono\">{_e(fmtf(st.get('min')))}</td>"
                f"<td class=\"mono\">{_e(fmtf(st.get('max')))}</td>"
                "</tr>"
            )

        num_table = (
            "<div class=\"table-scroll\">"
            "<table class=\"table small\">"
            "<thead><tr><th>numeric col</th><th>count</th><th>mean</th><th>min</th><th>max</th></tr></thead>"
            f"<tbody>{''.join(num_table_rows) if num_table_rows else '<tr><td colspan=\"5\" class=\"muted\">No numeric columns detected.</td></tr>'}</tbody>"
            "</table>"
            "</div>"
        )

    # Preview rows (default: 10 data rows)
    preview_rows_s = (q.get("preview_rows") or ["10"])[0].strip()
    try:
        preview_rows = max(1, min(500, int(preview_rows_s)))
    except Exception:
        preview_rows = 10

    # Preview
    preview_html = _render_csv_preview(csv_path, max_rows=preview_rows, run=run)

    # Preview controls (no JS)
    self_href = f"/csv/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/{_e(csv_name)}"
    preview_controls = (
        "<div class=\"sub\">Preview rows: "
        f"<a href=\"{self_href}?preview_rows=10\">10</a> "
        f"<a href=\"{self_href}?preview_rows=50\">50</a> "
        f"<a href=\"{self_href}?preview_rows=200\">200</a> "
        f"(current={_e(preview_rows)})"
        "</div>"
    )

    # Plots associated with this CSV
    plot_html = "<div class=\"muted\">No plots for this CSV.</div>"
    plot_paths: List[Path] = []
    if not preview_only:
        if csv_name == "token_len_scaling.csv":
            plot_paths = [run.analysis_dir / "plot_token_len_scaling.png"]
        elif csv_name == "batch_size_scaling.csv":
            plot_paths = [run.analysis_dir / "plot_batch_size_scaling.png"]
        elif csv_name == "cpu_scaling.csv":
            plot_paths = [run.analysis_dir / "plot_cpu_scaling.png"]
        elif csv_name == "kv_scaling.csv":
            plot_paths = [run.analysis_dir / "plot_kv_cap_scaling.png"]

    cards = []
    for p in plot_paths:
        if not p.exists():
            continue
        rel = p.relative_to(run.run_dir).as_posix()
        src = f"/raw/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/{_e(rel)}"
        cards.append(
            "<div class=\"plot\">"
            f"<div class=\"plot-title\">{_e(p.name)}</div>"
            f"<a href=\"{src}\" target=\"_blank\"><img src=\"{src}\" alt=\"{_e(p.name)}\" /></a>"
            "</div>"
        )
    if cards:
        plot_html = f"<div class=\"plots\">{''.join(cards)}</div>"

    # PPT-ready summary for scaling CSVs
    is_scaling = csv_name in {"token_len_scaling.csv", "batch_size_scaling.csv", "cpu_scaling.csv", "kv_scaling.csv"}
    ppt_summary_html = ""
    if is_scaling:
        ppt_summary_html = _render_scaling_csv_ppt_summary(csv_path)

    # Download link
    rel_csv = csv_path.relative_to(run.run_dir).as_posix()
    raw = f"/raw/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/{_e(rel_csv)}"
    back = f"/run/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}"


    title = f"{run.ref.task} / {run.ref.suite} / {run.ref.run_id} :: {csv_name}"
    if preview_only:
        body = f"""
<section class="card">
  <div class="breadcrumbs">
    <a href="/">Home</a>
    <span class="sep">/</span>
    <a href="{back}">{_e(run.ref.task)} / {_e(run.ref.suite)} / {_e(run.ref.run_id)}</a>
    <span class="sep">/</span>
    <span class="mono">{_e(csv_name)}</span>
  </div>
  <h1>CSV</h1>
  <div class="sub">file: <span class="mono">{_e(csv_name)}</span> • <a href="{raw}" target="_blank">Download</a></div>
  <div class="sub">This page intentionally shows preview only (no summary/plots) to keep the job navigation clean.</div>
</section>

<section class="card">
  <h2>Preview</h2>
    {preview_controls}
  {preview_html}
</section>
"""
    else:
        body = f"""
<section class="card">
  <div class="breadcrumbs">
    <a href="/">Home</a>
    <span class="sep">/</span>
    <a href="{back}">{_e(run.ref.task)} / {_e(run.ref.suite)} / {_e(run.ref.run_id)}</a>
    <span class="sep">/</span>
    <span class="mono">{_e(csv_name)}</span>
  </div>
  <h1>CSV</h1>
  <div class="sub">file: <span class="mono">{_e(csv_name)}</span> • rows~{_e(num_rows)} • cols~{_e(num_cols)} • <a href="{raw}" target="_blank">Download</a></div>
</section>

<section class="card">
    <h2>Summary</h2>
    {('<div class="sub">PPT 一页总结</div>' + ppt_summary_html + '<div class="sub">关键图</div>' + plot_html) if is_scaling else num_table}
</section>

{'' if is_scaling else f'''<section class="card">\n  <h2>Plots</h2>\n  {plot_html}\n</section>'''}

<section class="card">
  <h2>Preview</h2>
    {preview_controls}
  {preview_html}
</section>
"""
    return _html_page(title, body)


def _render_job_detail(scale_test_root: Path, run: RunInfo, job_name: str, q: Dict[str, List[str]]) -> str:
    # Ensure analysis artifacts exist.
    required = [run.analysis_dir / "emon_socket_metrics.csv", run.analysis_dir / "emon_job_pies_manifest.json"]
    _maybe_autogen_analysis(scale_test_root=scale_test_root, run=run, required=required)

    csv_path = (run.analysis_dir / "emon_socket_metrics.csv").resolve()
    if not _is_within(csv_path, run.analysis_dir) or not csv_path.exists():
        return _html_page("Job not found", "<section class=\"card\"><h1>emon_socket_metrics.csv not found</h1></section>")

    # Load row for this job.
    text = _read_text(csv_path, max_bytes=20_000_000)
    reader = csv.DictReader(io.StringIO(text))
    row: Optional[Dict[str, str]] = None
    for r in reader:
        if (r.get("job_name") or "").strip() == job_name:
            row = {k: ("" if v is None else str(v)) for k, v in r.items()}
            break

    back_run = f"/run/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}"
    back_csv = f"/csv/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/emon_socket_metrics.csv"
    if row is None:
        body_nf = f"""
<section class="card">
  <div class="breadcrumbs">
    <a href="/">Home</a>
    <span class="sep">/</span>
    <a href="{back_run}">{_e(run.ref.task)} / {_e(run.ref.suite)} / {_e(run.ref.run_id)}</a>
    <span class="sep">/</span>
    <a href="{back_csv}">emon_socket_metrics.csv</a>
    <span class="sep">/</span>
    <span class="mono">{_e(job_name)}</span>
  </div>
  <h1>Job</h1>
  <div class="warn mono">Job not found in emon_socket_metrics.csv</div>
</section>
"""
        return _html_page("Job not found", body_nf)

    # Sockets present.
    socket_keys = [k for k in row.keys() if k.startswith("socket_") and "__" in k]
    sockets = sorted({k.split("__", 1)[0] for k in socket_keys}, key=lambda s: (len(s), s))

    # Load pie manifest.
    manifest: Dict[str, Any] = {}
    man_path = (run.analysis_dir / "emon_job_pies_manifest.json").resolve()
    if _is_within(man_path, run.analysis_dir) and man_path.exists():
        try:
            manifest = json.loads(_read_text(man_path, max_bytes=8_000_000))
        except Exception:
            manifest = {}

    pies: Dict[str, str] = {}
    try:
        rec = manifest.get(job_name) or {}
        pies = rec.get("pies") or {}
    except Exception:
        pies = {}

    # Summary.xlsx download link if available.
    xlsx_dl = ""
    emon_xlsx = (row.get("emon_summary_xlsx") or "").strip()
    if emon_xlsx:
        try:
            p = Path(emon_xlsx).resolve()
            if _is_within(p, run.run_dir) and p.exists() and p.is_file():
                rel = p.relative_to(run.run_dir).as_posix()
                dl = quote(f"{job_name}_summary.xlsx")
                href = f"/raw/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/{_e(rel)}?download_name={dl}"
                xlsx_dl = f'<a class="btn" href="{href}" target="_blank">Download summary.xlsx</a>'
        except Exception:
            pass

    # Metric list (as requested).
    metrics = [
        "metric_CPU operating frequency (in GHz)",
        "metric_uncore frequency GHz",
        "metric_DDR data rate (MT/sec)",
        "metric_UPI speed - GT/s",
        "metric_CPU utilization %",
        "metric_CPU utilization% in kernel mode",
        "metric_CPI",
        "metric_kernel_CPI",
        "metric_core IPC",
        "metric_package power (watts)",
        "metric_L1D MPI (includes data+rfo w/ prefetches)",
        "metric_L2 MPI (includes code+data+rfo w/ prefetches)",
        "metric_LLC MPI (includes code+data+rfo w/ prefetches)",
        "metric_NUMA %_Reads addressed to local DRAM",
        "metric_NUMA %_Reads addressed to remote DRAM",
        "metric_memory bandwidth read (MB/sec)",
        "metric_memory bandwidth write (MB/sec)",
        "metric_memory bandwidth total (MB/sec)",
        "metric_core c6 residency %",
        "metric_package c6 residency %",
        "metric_package c2 residency %",
        "metric_TMA_Frontend_Bound(%)",
        "metric_TMA_Bad_Speculation(%)",
        "metric_TMA_Retiring(%)",
        "metric_TMA_Backend_Bound(%)",
        "metric_TMA_..Memory_Bound(%)",
        "metric_TMA_....L1_Bound(%)",
        "metric_TMA_....L2_Bound(%)",
        "metric_TMA_....L3_Bound(%)",
        "metric_TMA_....DRAM_Bound(%)",
        "metric_TMA_....Store_Bound(%)",
        "metric_TMA_..Core_Bound(%)",
    ]

    def _fmt_metric(metric: str, v: str) -> str:
        st = (v or "").strip()
        if not st or st.lower() in {"nan", "none"}:
            return ""
        try:
            x = float(st)
        except Exception:
            return _e(st)
        m = metric.lower()
        is_pct = "(%)" in metric or "utilization" in m or "residency" in m or "tma_" in m
        if is_pct:
            return _e(f"{x:.2f}")
        if abs(x) >= 1000:
            return _e(f"{x:.0f}")
        if abs(x) >= 100:
            return _e(f"{x:.2f}")
        return _e(f"{x:.3f}")

    # Metric table.
    head = "<tr><th>metric</th>" + "".join(f"<th>{_e(s)}</th>" for s in sockets) + "</tr>"
    trs: List[str] = []
    for m in metrics:
        tds = [f"<td class=\"mono\">{_e(m)}</td>"]
        for s in sockets:
            key = f"{s}__{m}"
            tds.append(f"<td class=\"mono\">{_fmt_metric(m, row.get(key, ''))}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    metric_table = (
        '<div class="table-scroll">'
        '<table class="table small preview-table">'
        f"<thead>{head}</thead><tbody>{''.join(trs)}</tbody>"
        "</table>"
        "</div>"
    )

    # Pie chart cards.
    pie_cards: List[str] = []
    for s in sockets:
        rel_png = str(pies.get(s) or "").strip()
        if not rel_png:
            continue
        try:
            p = (run.run_dir / rel_png).resolve()
            if not _is_within(p, run.run_dir) or not p.exists():
                continue
            rel = p.relative_to(run.run_dir).as_posix()
            src = f"/raw/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/{_e(rel)}"
            pie_cards.append(
                "<div class=\"plot\">"
                f"<div class=\"plot-title\">{_e(s)} TMA pie</div>"
                f"<a href=\"{src}\" target=\"_blank\"><img src=\"{src}\" alt=\"{_e(s)}\" /></a>"
                "</div>"
            )
        except Exception:
            continue
    pies_html = f"<div class=\"plots\">{''.join(pie_cards)}</div>" if pie_cards else '<div class="muted">No per-job TMA pie images found (try re-running analyze_run.py).</div>'

    # A few meta fields.
    meta_rows = []
    for k in ["variant", "resource_cpu", "resource_cpu_count", "sglang_max_total_tokens", "batch_size", "token_len"]:
        if k in row:
            meta_rows.append((k, row.get(k, "")))
    meta_trs = "".join(f"<tr><td class=\"mono\">{_e(k)}</td><td class=\"mono\">{_e(v)}</td></tr>" for k, v in meta_rows)
    meta_table = (
        '<div class="table-scroll">'
        '<table class="table small">'
        '<thead><tr><th>key</th><th>value</th></tr></thead>'
        f"<tbody>{meta_trs}</tbody></table></div>"
        if meta_rows
        else ""
    )

    title = f"{run.ref.task} / {run.ref.suite} / {run.ref.run_id} :: job {job_name}"
    body = f"""
<section class="card">
  <div class="breadcrumbs">
    <a href="/">Home</a>
    <span class="sep">/</span>
    <a href="{back_run}">{_e(run.ref.task)} / {_e(run.ref.suite)} / {_e(run.ref.run_id)}</a>
    <span class="sep">/</span>
    <a href="{back_csv}">emon_socket_metrics.csv</a>
    <span class="sep">/</span>
    <span class="mono">{_e(job_name)}</span>
  </div>
  <h1>Job</h1>
  <div class="sub">job_name: <span class="mono">{_e(job_name)}</span></div>
  <div class="toolbar">{xlsx_dl}</div>
  {meta_table}
</section>

<section class="card">
  <h2>Socket TMA pies</h2>
  {pies_html}
</section>

<section class="card">
  <h2>Socket metrics</h2>
  {metric_table}
</section>
"""
    return _html_page(title, body)


def _render_csv_preview(path: Path, max_rows: int = 200, *, run: Optional[RunInfo] = None) -> str:
    # Special-case: make job_name clickable to download summary.xlsx.
    if path.name == "emon_socket_metrics.csv" and run is not None:
        text = _read_text(path, max_bytes=2_000_000)
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            return '<div class="muted">Empty CSV.</div>'

        fieldnames = list(reader.fieldnames)
        rows: List[Dict[str, str]] = []
        for i, r in enumerate(reader):
            rows.append({k: ("" if v is None else str(v)) for k, v in r.items()})
            if i + 1 >= max_rows:
                break

        def _td_html(cell_html: str, *, is_head: bool = False, cls: str = "") -> str:
            tag = "th" if is_head else "td"
            cls_attr = f' class="{cls}"' if cls else ""
            return f"<{tag}{cls_attr}>{cell_html}</{tag}>"

        def _td_text(cell: str, *, is_head: bool = False) -> str:
            tag = "th" if is_head else "td"
            cls = "mono" if (len(cell) < 64 and any(ch.isdigit() for ch in cell)) else ""
            return f"<{tag} class=\"{cls}\">{_e(cell)}</{tag}>"

        def _summary_href(emon_path_s: str, job_name: str) -> Optional[str]:
            s = (emon_path_s or "").strip()
            if not s:
                return None
            try:
                p = Path(s).resolve()
                if not _is_within(p, run.run_dir) or not p.exists() or not p.is_file():
                    return None
                rel = p.relative_to(run.run_dir).as_posix()
                dl = quote(f"{job_name}_summary.xlsx")
                return f"/raw/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/{_e(rel)}?download_name={dl}"
            except Exception:
                return None

        def _job_href(job_name: str) -> str:
            return f"/job/{_e(run.ref.task)}/{_e(run.ref.suite)}/{_e(run.ref.run_id)}/{quote(job_name)}"

        thead = "<tr>" + "".join(_td_text(c, is_head=True) for c in fieldnames) + "</tr>"
        body_trs: List[str] = []
        for r in rows:
            tds: List[str] = []
            job = r.get("job_name", "")
            emon_xlsx = r.get("emon_summary_xlsx", "")
            href = _summary_href(emon_xlsx, job)
            for c in fieldnames:
                v = r.get(c, "")
                if c == "job_name" and job:
                    tds.append(_td_html(f'<a href="{_job_href(job)}">{_e(job)}</a>', cls="mono"))
                elif c == "emon_summary_xlsx" and href:
                    tds.append(_td_html(f'<a href="{href}" target="_blank">summary.xlsx</a>', cls="mono"))
                else:
                    tds.append(_td_text(v, is_head=False))
            body_trs.append("<tr>" + "".join(tds) + "</tr>")

        tbody = "".join(body_trs)
        return (
            f'<div class="sub">Preview: <span class="mono">{_e(path.name)}</span> (first {len(rows)} rows) — click <span class="mono">job_name</span> to open the job page</div>'
            '<div class="table-scroll preview-scroll">'
            f'<table class="table small preview-table"><thead>{thead}</thead><tbody>{tbody}</tbody></table>'
            "</div>"
        )

    # Default: stream the CSV and escape for HTML.
    buf = io.StringIO(_read_text(path, max_bytes=2_000_000))
    reader2 = csv.reader(buf)
    header = next(reader2, None)
    if not header:
        return '<div class="muted">Empty CSV.</div>'
    body_rows: List[List[str]] = []
    for i, row in enumerate(reader2):
        body_rows.append([str(c) for c in row])
        if i + 1 >= max_rows:
            break

    def td(cell: str, is_head: bool = False) -> str:
        tag = "th" if is_head else "td"
        cls = "mono" if (len(cell) < 64 and any(ch.isdigit() for ch in cell)) else ""
        return f"<{tag} class=\"{cls}\">{_e(cell)}</{tag}>"

    thead2 = "<tr>" + "".join(td(c, is_head=True) for c in header) + "</tr>"
    tbody2 = "".join("<tr>" + "".join(td(c, is_head=False) for c in row) + "</tr>" for row in body_rows)
    return (
        f'<div class="sub">Preview: <span class="mono">{_e(path.name)}</span> (first {len(body_rows)} rows)</div>'
        '<div class="table-scroll preview-scroll">'
        f'<table class="table small preview-table"><thead>{thead2}</thead><tbody>{tbody2}</tbody></table>'
        "</div>"
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    server_version = "scale-test-web/0.1"

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._do_GET()
        except Exception as e:
            self._send(500, f"Internal error: {e}\n", content_type="text/plain")

    def _do_GET(self) -> None:
        srv: "WebServer" = self.server  # type: ignore[assignment]
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        q = parse_qs(parsed.query)

        if path == "/" or path == "":
            runs = discover_runs(srv.scale_test_root)
            html_text = _render_home(srv.scale_test_root, runs, q)
            self._send(200, html_text, content_type="text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            file_path = (WEB_DIR / "static" / rel).resolve()
            if not _is_within(file_path, WEB_DIR / "static") or not file_path.exists():
                self._send(404, "Not Found\n", content_type="text/plain")
                return
            self._send_file(file_path)
            return

        if path.startswith("/run/"):
            parts = [p for p in path.split("/") if p]
            if len(parts) != 4:
                self._send(404, "Not Found\n", content_type="text/plain")
                return
            _, task, suite, run_id = parts
            run = srv.get_run(task, suite, run_id)
            if run is None:
                self._send(404, "Run not found\n", content_type="text/plain")
                return
            html_text = _render_run(srv.scale_test_root, run, q)
            self._send(200, html_text, content_type="text/html; charset=utf-8")
            return

        if path.startswith("/csv/"):
            parts = [p for p in path.split("/") if p]
            if len(parts) != 5:
                self._send(404, "Not Found\n", content_type="text/plain")
                return
            _, task, suite, run_id, csv_name = parts
            run = srv.get_run(task, suite, run_id)
            if run is None:
                self._send(404, "Run not found\n", content_type="text/plain")
                return
            html_text = _render_csv_detail(srv.scale_test_root, run, csv_name, q)
            self._send(200, html_text, content_type="text/html; charset=utf-8")
            return

        if path.startswith("/job/"):
            parts = [p for p in path.split("/") if p]
            if len(parts) != 5:
                self._send(404, "Not Found\n", content_type="text/plain")
                return
            _, task, suite, run_id, job_name = parts
            run = srv.get_run(task, suite, run_id)
            if run is None:
                self._send(404, "Run not found\n", content_type="text/plain")
                return
            html_text = _render_job_detail(srv.scale_test_root, run, job_name, q)
            self._send(200, html_text, content_type="text/html; charset=utf-8")
            return

        if path.startswith("/raw/"):
            parts = [p for p in path.split("/") if p]
            if len(parts) < 5:
                self._send(404, "Not Found\n", content_type="text/plain")
                return
            _, task, suite, run_id, *rest = parts
            run = srv.get_run(task, suite, run_id)
            if run is None:
                self._send(404, "Run not found\n", content_type="text/plain")
                return
            rel = "/".join(rest)
            file_path = (run.run_dir / rel).resolve()
            if not _is_within(file_path, run.run_dir) or not file_path.exists() or not file_path.is_file():
                self._send(404, "Not Found\n", content_type="text/plain")
                return
            dl_name = (q.get("download_name") or q.get("filename") or [""])[0].strip()
            if dl_name:
                # If the client asks for a custom name, always treat as attachment.
                # Keep extension consistent with served content when possible.
                ext = file_path.suffix if file_path.suffix else None
                safe_name = _sanitize_download_name(dl_name, default_ext=ext)
                self._send_file(file_path, download_name=safe_name)
            else:
                self._send_file(file_path)
            return

        self._send(404, "Not Found\n", content_type="text/plain")

    def _send(self, code: int, body: str, *, content_type: str) -> None:
        data = body.encode("utf-8", errors="replace")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, *, download_name: Optional[str] = None) -> None:
        ctype, _ = mimetypes.guess_type(str(path))
        if not ctype:
            ctype = "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if download_name:
            safe = _sanitize_download_name(download_name, default_ext=path.suffix)
            self.send_header("Content-Disposition", f'attachment; filename="{safe}"')
        # allow browser caching for static-ish artifacts
        self.send_header("Cache-Control", "public, max-age=60")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep logs concise.
        syslog = os.environ.get("SCALE_TEST_WEB_LOG", "1")
        if syslog != "0":
            super().log_message(fmt, *args)


class WebServer(ThreadingHTTPServer):
    def __init__(self, server_address: Tuple[str, int], handler_cls: Any, *, scale_test_root: Path):
        super().__init__(server_address, handler_cls)
        self.scale_test_root = scale_test_root

    def get_run(self, task: str, suite: str, run_id: str) -> Optional[RunInfo]:
        run_dir = (self.scale_test_root / task / "result" / suite / run_id).resolve()
        analysis_dir = run_dir / "analysis"
        if not _is_within(run_dir, self.scale_test_root) or not analysis_dir.exists():
            return None
        try:
            mtime = run_dir.stat().st_mtime
        except Exception:
            mtime = 0.0
        return RunInfo(ref=RunRef(task=task, suite=suite, run_id=run_id), run_dir=run_dir, analysis_dir=analysis_dir, mtime=mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description="Scale-test results web UI")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--scale-test-root",
        default=str(SCALE_TEST_ROOT_DEFAULT),
        help="Path to scripts/scale-test (default: inferred from this file)",
    )
    args = ap.parse_args()

    scale_test_root = Path(args.scale_test_root).resolve()
    if not scale_test_root.exists():
        raise SystemExit(f"scale-test root not found: {scale_test_root}")

    # Warm up mimetypes.
    mimetypes.init()

    httpd = WebServer((args.host, int(args.port)), Handler, scale_test_root=scale_test_root)
    print(f"[ok] Serving: http://{args.host}:{args.port}/")
    print(f"[ok] Root: {scale_test_root}")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
