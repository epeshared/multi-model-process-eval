#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_METRICS: List[str] = []


DEFAULT_SOCKET_VIEW_METRICS = [
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


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _try_import() -> Tuple[Any, Optional[Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        _eprint("ERROR: pandas is required for analysis")
        _eprint(f"Import error: {e}")
        _eprint("Try: python -m pip install pandas")
        raise

    plt = None
    try:
        import matplotlib  # type: ignore

        # Headless-safe backend (common on servers / SSH).
        matplotlib.use("Agg", force=True)  # type: ignore[attr-defined]
        import matplotlib.pyplot as _plt  # type: ignore

        plt = _plt
    except Exception as e:
        _eprint("WARN: matplotlib not available; skipping plots")
        _eprint(f"Import error: {e}")

    return pd, plt


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int_sort_key(x: Any) -> Tuple[int, str]:
    s = str(x).strip()
    try:
        return int(s), s
    except Exception:
        return 10**18, s


def _iter_variant_pairs(df: Any) -> Iterable[Tuple[str, str]]:
    # (cpu_expr, kv_cap)
    cols = set(df.columns)
    if "resource_cpu" not in cols or "sglang_max_total_tokens" not in cols:
        return []
    pairs = (
        df[["resource_cpu", "sglang_max_total_tokens"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    return [(str(a).strip(), str(b).strip()) for a, b in pairs]


def _flatten_pivot_columns(columns: Any, suffix: str) -> List[str]:
    out: List[str] = []
    for col in columns:
        try:
            tl, bs = col  # type: ignore[misc]
        except Exception:
            out.append(f"{col}_{suffix}")
            continue
        out.append(f"tok{int(tl)}_bs{int(bs)}_{suffix}")
    return out


def _write_csv(df: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _sanitize_header(s: Any) -> str:
    out = str(s).strip()
    out = out.replace(" ", "_")
    return out


def _norm_metric_name(s: Any) -> str:
    """Normalize metric names to be resilient to minor formatting diffs.

    - lower
    - remove spaces and underscores
    - collapse repeated dots
    """
    t = str(s).strip().lower()
    t = t.replace(" ", "")
    t = t.replace("_", "")
    # keep percent and parentheses since they help disambiguate
    while ".." in t:
        t = t.replace("..", ".")
    return t


def _fmt_metric_value(metric: str, v: Any) -> str:
    x = _to_float(v)
    if x is None:
        return ""
    m = str(metric)
    is_pct = "(%)" in m or "utilization" in m.lower() or "residency" in m.lower() or "tma_" in m.lower()
    if is_pct:
        return f"{x:.2f}%" if x < 1 and "(%)" in m else f"{x:.2f}"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    if abs(x) >= 100:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sanitize_job_id(job_name: str, *, max_len: int = 80) -> str:
    """Create a filesystem-safe, mostly-human-readable id for a job.

    We append a short hash to avoid collisions when job names are long/similar.
    """
    raw = str(job_name or "").strip()
    if not raw:
        raw = "job"
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {".", "-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("._-") or "job"
    if len(s) > max_len:
        s = s[:max_len]
    h = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{s}__{h}"


def _guess_auto_test_stdout_log(run_dir: Path, variant: str) -> str:
    """Best-effort locate auto_test_stdout.log for a variant.

    Layouts we support:
    - run_dir/<variant>/auto_test_stdout.log               (older expected)
    - run_dir/auto_test_stdout.log                         (single-variant)
    - run_dir/hosts/<host>/auto_test_stdout.log            (scale-test multi-host)
    """

    v = str(variant or "").strip()
    if v:
        p1 = (run_dir / v / "auto_test_stdout.log").resolve()
        if p1.exists():
            return str(p1)

    p2 = (run_dir / "auto_test_stdout.log").resolve()
    if p2.exists():
        return str(p2)

    hosts_dir = run_dir / "hosts"
    if hosts_dir.exists() and hosts_dir.is_dir():
        try:
            for p in sorted(hosts_dir.glob("*/auto_test_stdout.log")):
                if p.exists():
                    return str(p.resolve())
        except Exception:
            pass

    # Fallback to the historical default.
    if v:
        return str((run_dir / v / "auto_test_stdout.log").resolve())
    return str((run_dir / "auto_test_stdout.log").resolve())


def _render_run_summary_html(
    *,
    run_dir: Path,
    out_dir: Path,
    df_jobs: Any,
    failed_variants: Any,
    df_socket: Any,
    wanted_metrics: List[str],
) -> str:
    pd = df_jobs.__class__  # dummy

    def _h3(title: str) -> str:
        return f"<h3>{html.escape(title)}</h3>"

    def _kv_table(title: str, rows: List[Tuple[str, str]]) -> str:
        trs = "".join(
            f"<tr><td class=\"mono\">{html.escape(k)}</td><td class=\"mono\">{html.escape(v)}</td></tr>" for k, v in rows
        )
        return (
            f"{_h3(title)}"
            "<div class=\"table-scroll\">"
            "<table class=\"table small\">"
            "<thead><tr><th>key</th><th>value</th></tr></thead>"
            f"<tbody>{trs}</tbody>"
            "</table>"
            "</div>"
        )

    def _impact_level_from_ratio(r: float) -> Tuple[str, str]:
        # r ~= (max/min) within-group variability
        if r >= 3.0:
            return ("高", "strong")
        if r >= 1.5:
            return ("中", "warn")
        return ("低", "ok")

    def _impact_level_from_eff(e: float) -> Tuple[str, str]:
        # higher is better
        if e >= 0.80:
            return ("好", "ok")
        if e >= 0.60:
            return ("一般", "warn")
        return ("较差", "strong")

    def _read_scaling_csv(name: str) -> Any:
        try:
            p = out_dir / name
            if not p.exists():
                return None
            return __import__("pandas").read_csv(p)  # type: ignore
        except Exception:
            return None

    def _tps_pivot_table_html() -> str:
        """Render a compact TPS pivot table (rows=batch_size, cols=token_len)."""

        try:
            pd_mod = __import__("pandas")
        except Exception:
            return '<div class="muted">No pandas; cannot build TPS pivot table.</div>'

        if df_jobs is None or getattr(df_jobs, "empty", True):
            return '<div class="muted">No job data.</div>'

        need = {"batch_size", "token_len", "tps"}
        cols = set(getattr(df_jobs, "columns", []))
        if not need.issubset(cols):
            return '<div class="muted">Missing required columns for TPS pivot.</div>'

        d = df_jobs.dropna(subset=["batch_size", "token_len", "tps"]).copy()
        if getattr(d, "empty", True):
            return '<div class="muted">No non-empty rows for TPS pivot.</div>'

        # Prefer using derived fields when present.
        if "cpu_count" in d.columns:
            d["cpu_count"] = pd_mod.to_numeric(d["cpu_count"], errors="coerce")
        if "kv_cap" in d.columns:
            d["kv_cap"] = pd_mod.to_numeric(d["kv_cap"], errors="coerce")

        # If multiple slices exist (cpu/kv/model), pick the most common slice.
        slice_cols = [
            c
            for c in ["resource_cpu", "cpu_count", "kv_cap", "sglang_max_total_tokens"]
            if c in d.columns and d[c].nunique(dropna=False) > 1
        ]
        slice_note = ""
        if slice_cols:
            try:
                counts = d.groupby(slice_cols, dropna=False).size().reset_index(name="n")
                counts = counts.sort_values(["n"], ascending=False, kind="mergesort")
                best = counts.iloc[0]
                mask = None
                for c in slice_cols:
                    v = best[c]
                    if mask is None:
                        mask = d[c].isna() if pd_mod.isna(v) else (d[c] == v)
                    else:
                        mask = mask & (d[c].isna() if pd_mod.isna(v) else (d[c] == v))
                if mask is not None:
                    d = d[mask]
                picked = ", ".join(f"{c}={best[c]}" for c in slice_cols)
                slice_note = f'<div class="sub">slice: <span class="mono">{html.escape(picked)}</span></div>'
            except Exception:
                slice_note = ""

        pv = d.pivot_table(index="batch_size", columns="token_len", values="tps", aggfunc="mean")
        if getattr(pv, "empty", True):
            return '<div class="muted">Empty TPS pivot.</div>'

        # Sort axes numerically.
        try:
            pv = pv.sort_index(axis=0)
        except Exception:
            pass
        try:
            pv = pv.reindex(sorted(list(pv.columns), key=lambda x: float(x)), axis=1)
        except Exception:
            pass

        def fmt(v: Any) -> str:
            try:
                if pd_mod.isna(v):
                    return ""
            except Exception:
                pass
            try:
                return f"{float(v):.3f}"
            except Exception:
                return str(v)

        ths = "".join(f"<th class=\"mono\">tok{html.escape(str(int(float(c))) if float(c).is_integer() else str(c))}</th>" for c in pv.columns)
        head = f"<thead><tr><th>batch_size</th>{ths}</tr></thead>"
        body_rows: List[str] = []
        for bs in pv.index:
            tds = "".join(f"<td class=\"mono\">{html.escape(fmt(pv.loc[bs, c]))}</td>" for c in pv.columns)
            body_rows.append(f"<tr><td class=\"mono\">bs={html.escape(fmt(bs))}</td>{tds}</tr>")
        table = (
            "<div class=\"table-scroll\">"
            "<table class=\"table small\">"
            f"{head}<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
            "</div>"
        )
        return (
            "<h3>TPS pivot（行=batch_size，列=token_len）</h3>"
            "<div class=\"sub\">统计口径：对同一 token_len×batch_size 的 TPS 做 mean（必要时先选取最常见 slice）。</div>"
            + slice_note
            + table
        )

    def _to_num(pd_mod: Any, s: Any) -> Any:
        try:
            return pd_mod.to_numeric(s, errors="coerce")
        except Exception:
            return s

    def _scaling_summary(
        *,
        title: str,
        df: Any,
        x: str,
        group_cols: List[str],
        y: str = "tps",
        corr_y: str | None = None,
    ) -> Tuple[str, Dict[str, float]]:
        stats_out: Dict[str, float] = {}
        if df is None or getattr(df, "empty", True):
            return (f"<h3>{html.escape(title)}</h3><div class=\"muted\">No data.</div>", stats_out)
        pd_mod = __import__("pandas")

        cols = set(getattr(df, "columns", []))
        need = [x, y] + [c for c in group_cols]
        if not all(c in cols for c in need):
            return (
                f"<h3>{html.escape(title)}</h3>"
                f"<div class=\"muted\">Missing required columns: {html.escape(', '.join([c for c in need if c not in cols]))}</div>"
            , stats_out)

        # IMPORTANT: do NOT drop rows based on group columns. Some dimensions
        # (e.g. kv_cap parsed from sglang_max_total_tokens="auto") become NaN;
        # dropping NaN here would erase the entire run from the summary.
        d = df.dropna(subset=[x, y]).copy()
        if getattr(d, "empty", True):
            return (f"<h3>{html.escape(title)}</h3><div class=\"muted\">No non-empty rows.</div>", stats_out)
        d[x] = _to_num(pd_mod, d[x])
        d[y] = _to_num(pd_mod, d[y])

        # Per-group best-x (argmax y) and variability across x.
        best_x: List[float] = []
        ratios: List[float] = []
        cors: List[float] = []
        n_groups = 0
        for _, g in d.groupby(group_cols, dropna=False):
            n_groups += 1
            g2 = g.dropna(subset=[x, y]).copy()
            if g2.empty or g2[x].nunique() < 2:
                continue
            try:
                r = g2.loc[g2[y].idxmax()]
                best_x.append(float(r[x]))
            except Exception:
                pass
            try:
                ymin = float(g2[y].min())
                ymax = float(g2[y].max())
                if ymin > 0:
                    ratios.append(ymax / ymin)
            except Exception:
                pass
            if corr_y is not None and corr_y in cols:
                try:
                    g3 = g2.dropna(subset=[x, corr_y]).copy()
                    g3[corr_y] = _to_num(pd_mod, g3[corr_y])
                    if not g3.empty and g3[x].nunique() >= 2:
                        corr = g3[[x, corr_y]].corr(numeric_only=True).iloc[0, 1]
                        if corr == corr:
                            cors.append(float(corr))
                except Exception:
                    pass

        def _q(vals: List[float], q: float) -> str:
            if not vals:
                return ""
            try:
                return f"{float(pd_mod.Series(vals).quantile(q)):.2f}"
            except Exception:
                return ""

        items: List[str] = []
        rows_n = float(len(d))
        groups_n = float(n_groups)
        stats_out["rows"] = rows_n
        stats_out["groups"] = groups_n

        med_best = float(pd_mod.Series(best_x).median()) if best_x else float("nan")
        med_ratio = float(pd_mod.Series(ratios).median()) if ratios else float("nan")
        p90_ratio = float(pd_mod.Series(ratios).quantile(0.90)) if ratios else float("nan")
        med_corr = float(pd_mod.Series(cors).median()) if cors else float("nan")
        if med_best == med_best:
            stats_out["median_best_x"] = med_best
        if med_ratio == med_ratio:
            stats_out["median_ratio"] = med_ratio
        if p90_ratio == p90_ratio:
            stats_out["p90_ratio"] = p90_ratio
        if med_corr == med_corr:
            stats_out["median_corr"] = med_corr

        impact_label = ""
        impact_class = ""
        if med_ratio == med_ratio:
            impact_label, impact_class = _impact_level_from_ratio(med_ratio)

        # PPT-ready narrative
        takeaway = ""
        if x == "batch_size":
            takeaway = "批量大小主要影响吞吐上限；本次 sweep 中整体影响" + ("较小" if (med_ratio == med_ratio and med_ratio < 1.5) else "较大") + "。"
        elif x == "kv_cap":
            takeaway = "KV cap 是关键瓶颈维度之一；不同 kv_cap 下 TPS 波动明显。"
        elif x == "token_len":
            takeaway = "token_len 改变会显著影响单位时间可处理的 token 数；通常 token_len 越大 tokens/sec 越低。"
        else:
            takeaway = f"{x} 会改变整体性能表现。"

        rec = ""
        if med_best == med_best:
            rec = f"建议优先选择 {x}≈{med_best:.0f}（按当前 sweep 的中位最优点）。"
        if x == "token_len" and (med_corr == med_corr):
            if med_corr <= -0.3:
                rec = "在固定 cpu/kv/bs 下，token_len 增大时 tokens/sec 往往下降；建议对齐目标场景的 token_len 做对比，避免用单一 token_len 代表所有场景。"
        if x == "kv_cap" and (med_ratio == med_ratio and med_ratio >= 1.5):
            rec = "KV cap 对 TPS 影响较大；建议优先确认不会被 KV 上限卡住（必要时提升 SGLANG_MAX_TOTAL_TOKENS / kv_cap）。"
        if x == "batch_size" and (med_ratio == med_ratio and med_ratio < 1.2):
            rec = "batch_size 对 TPS 的总体影响较小；可优先以延迟/内存压力为约束来选 batch_size。"

        items.append(f"<li><b>一句话结论</b>：{html.escape(takeaway)}</li>")
        if impact_label:
            items.append(f"<li><b>总体影响等级</b>：<span class=\"pill {impact_class}\">{html.escape(impact_label)}</span>（以组内 max/min 作为影响强度衡量）</li>")
        items.append(f"<li><b>统计口径</b>：基于 {html.escape(title)}（mean across repeats），rows={int(rows_n)}, groups={int(groups_n)}（按 {html.escape(','.join(group_cols))} 分组）</li>")
        if med_best == med_best:
            items.append(f"<li><b>中位最优点</b>：{html.escape(x)}≈<b>{med_best:.0f}</b>（argmax {html.escape(y)}）</li>")
        if med_ratio == med_ratio:
            items.append(f"<li><b>影响幅度</b>：组内 {html.escape(y)} 的 median(max/min)=<b>{med_ratio:.2f}×</b>（p90 {p90_ratio:.2f}×）</li>")
        if corr_y and (med_corr == med_corr):
            items.append(f"<li><b>趋势</b>：median corr({html.escape(x)}, {html.escape(corr_y)})=<b>{med_corr:.2f}</b></li>")
        if rec:
            items.append(f"<li><b>建议</b>：{html.escape(rec)}</li>")

        return (f"<h3>{html.escape(title)}</h3><ul>{''.join(items)}</ul>", stats_out)

    token_df = _read_scaling_csv("token_len_scaling.csv")
    bs_df = _read_scaling_csv("batch_size_scaling.csv")
    cpu_df = _read_scaling_csv("cpu_scaling.csv")
    kv_df = _read_scaling_csv("kv_scaling.csv")

    parts: List[str] = []
    parts.append('<div class="sub">Scale Test 总结（可直接放 PPT）：按四个 *_scaling.csv 给出总体影响与建议</div>')

    token_y = "tokens_per_sec" if token_df is not None and "tokens_per_sec" in getattr(token_df, "columns", []) else "tps"
    token_corr_y = "tokens_per_sec" if token_df is not None and "tokens_per_sec" in getattr(token_df, "columns", []) else None
    token_html, token_stats = _scaling_summary(
        title="token_len_scaling.csv",
        df=token_df,
        x="token_len",
        group_cols=["cpu_count", "kv_cap", "batch_size"],
        y=token_y,
        corr_y=token_corr_y,
    )
    bs_html, bs_stats = _scaling_summary(
        title="batch_size_scaling.csv",
        df=bs_df,
        x="batch_size",
        group_cols=["cpu_count", "kv_cap", "token_len"],
        y="tps",
        corr_y=None,
    )
    kv_html, kv_stats = _scaling_summary(
        title="kv_scaling.csv",
        df=kv_df,
        x="kv_cap",
        group_cols=["cpu_count", "batch_size", "token_len"],
        y="tps",
        corr_y=None,
    )

    # CPU scaling summary (speedup + efficiency)
    cpu_html = "<h3>cpu_scaling.csv</h3><div class=\"muted\">No data.</div>"
    cpu_stats: Dict[str, float] = {}
    try:
        if cpu_df is not None and not getattr(cpu_df, "empty", True):
            pd_mod = __import__("pandas")
            need = ["cpu_count", "tps", "kv_cap", "batch_size", "token_len"]
            if all(c in cpu_df.columns for c in need):
                d = cpu_df.dropna(subset=need).copy()
                d["cpu_count"] = pd_mod.to_numeric(d["cpu_count"], errors="coerce")
                d["tps"] = pd_mod.to_numeric(d["tps"], errors="coerce")
                speedups: List[float] = []
                effs: List[float] = []
                for _, g in d.groupby(["kv_cap", "batch_size", "token_len"], dropna=True):
                    g2 = g.sort_values("cpu_count")
                    if g2["cpu_count"].nunique() < 2:
                        continue
                    min_cpu = float(g2["cpu_count"].iloc[0])
                    max_cpu = float(g2["cpu_count"].iloc[-1])
                    tps_min = float(g2["tps"].iloc[0])
                    tps_max = float(g2["tps"].iloc[-1])
                    if min_cpu <= 0 or max_cpu <= 0 or tps_min <= 0:
                        continue
                    speedup = tps_max / tps_min
                    ideal = max_cpu / min_cpu
                    eff = speedup / ideal
                    speedups.append(speedup)
                    effs.append(eff)
                med_speed = float(pd_mod.Series(speedups).median()) if speedups else float("nan")
                med_eff = float(pd_mod.Series(effs).median()) if effs else float("nan")
                if med_speed == med_speed:
                    cpu_stats["median_speedup"] = med_speed
                if med_eff == med_eff:
                    cpu_stats["median_eff"] = med_eff

                lvl, cls = ("", "")
                if med_eff == med_eff:
                    lvl, cls = _impact_level_from_eff(med_eff)

                takeaway = "CPU 核数增加能提升吞吐，但总体呈次线性；需要关注并行效率。"
                rec = ""
                if med_eff == med_eff:
                    if med_eff >= 0.80:
                        rec = "CPU 扩展性较好；可优先用更多核数提升吞吐，注意 NUMA/绑核一致性。"
                    elif med_eff >= 0.60:
                        rec = "CPU 扩展性一般；建议优先排查线程/NUMA/内存带宽瓶颈，并结合 kv_cap 调优。"
                    else:
                        rec = "CPU 扩展性较差；优先优化瓶颈（kv_cap/带宽/调度），否则单纯加核收益有限。"

                items: List[str] = []
                items.append(f"<li><b>一句话结论</b>：{html.escape(takeaway)}</li>")
                if lvl:
                    items.append(f"<li><b>扩展性评价</b>：<span class=\"pill {cls}\">{html.escape(lvl)}</span>（以效率衡量）</li>")
                items.append(f"<li><b>统计口径</b>：基于 cpu_scaling.csv（mean across repeats），rows={len(d)}, groups={d.groupby(['kv_cap','batch_size','token_len']).ngroups}</li>")
                if med_speed == med_speed:
                    items.append(f"<li><b>中位加速比</b>：min→max cpu_count 的 median speedup=<b>{med_speed:.2f}×</b></li>")
                if med_eff == med_eff:
                    items.append(f"<li><b>中位并行效率</b>：median efficiency vs ideal linear=<b>{med_eff:.2f}</b></li>")
                if rec:
                    items.append(f"<li><b>建议</b>：{html.escape(rec)}</li>")
                cpu_html = f"<h3>cpu_scaling.csv</h3><ul>{''.join(items)}</ul>"
    except Exception:
        pass

    # Overall impact ranking (by median max/min ratio when available)
    # Always list key dimensions; show N/A if a dimension wasn't swept or lacks
    # enough in-group points.
    rank_src: List[Tuple[str, Dict[str, float]]] = [
        ("token_len", token_stats),
        ("batch_size", bs_stats),
        ("kv_cap", kv_stats),
    ]

    ranked: List[Tuple[str, float]] = []
    missing: List[str] = []
    for name, st in rank_src:
        r = st.get("median_ratio")
        try:
            rr = float(r) if r is not None else float("nan")
        except Exception:
            rr = float("nan")
        if rr == rr:
            ranked.append((name, rr))
        else:
            missing.append(name)

    ranked.sort(key=lambda t: t[1], reverse=True)

    lis2: List[str] = []
    for n, v in ranked:
        lis2.append(f"<li><span class=\"mono\">{html.escape(n)}</span>：median(max/min)=<b>{v:.2f}×</b></li>")
    for n in missing:
        lis2.append(
            f"<li><span class=\"mono\">{html.escape(n)}</span>：<span class=\"muted\">N/A（未 sweep 或有效点不足）</span></li>"
        )
    rank_html = f"<ul>{''.join(lis2)}</ul>" if lis2 else "<div class=\"muted\">No ranking available.</div>"

    parts.append(_h3("总体结论（影响强度排序）"))
    parts.append(
        "<div class=\"sub\">说明：以各维度在固定其它条件下的组内 TPS/throughput 波动（median max/min）衡量影响强度；CPU 另用 speedup/efficiency 描述。</div>"
    )
    parts.append(rank_html)

    # Add the TPS pivot table explicitly (commonly used in dashboards)
    parts.append(_tps_pivot_table_html())

    parts.append(token_html)
    parts.append(bs_html)
    parts.append(cpu_html)
    parts.append(kv_html)

    return "\n\n".join(parts)


def _extract_socket_view_from_xlsx(
    *,
    pd: Any,
    xlsx_path: str,
    wanted_metrics: List[str],
) -> Dict[str, Any]:
    """Return a flattened dict of socket-view metrics.

    Output columns look like: socket_0__metric_CPU operating frequency (in GHz)
    """
    p = Path(str(xlsx_path)).expanduser()
    if not p.exists():
        return {}
    try:
        df_sv = pd.read_excel(str(p), sheet_name="socket view")
    except Exception:
        return {}

    if df_sv is None or getattr(df_sv, "empty", True):
        return {}

    cols = list(df_sv.columns)
    if len(cols) < 2:
        return {}

    metric_col = cols[0]
    socket_cols = cols[1:]

    # Build a lookup by normalized metric name.
    tmp = df_sv[[metric_col] + socket_cols].copy()
    tmp[metric_col] = tmp[metric_col].astype(str).str.strip()
    tmp = tmp[tmp[metric_col].str.len() > 0]
    by_name = {str(n).strip().lower(): row for n, row in tmp.set_index(metric_col).iterrows()}

    out: Dict[str, Any] = {}
    for m in wanted_metrics:
        row = by_name.get(str(m).strip().lower())
        if row is None:
            continue
        for sc in socket_cols:
            key = f"{_sanitize_header(sc)}__{m}"
            out[key] = row.get(sc)
    return out


def _fmt_point_value(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    if math.isnan(x) or math.isinf(x):
        return ""
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def build_failed_variants(df_all: Any, run_dir: Path) -> Any:
    pd = df_all.__class__  # dummy to satisfy type checkers
    # overwritten by caller; here we assume pandas DataFrame.


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze a scale-test embedding run directory")
    ap.add_argument(
        "run_dir",
        help="Path to a scale-test run directory (the folder containing aggregate.csv)",
    )
    ap.add_argument(
        "--out-dir",
        default="analysis",
        help="Output subdir name under run_dir (default: analysis)",
    )
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="(deprecated) previously extracted emon_metrics.csv; no longer used",
    )
    ap.add_argument(
        "--socket-metrics",
        nargs="*",
        default=None,
        help="EMON socket-view metric keys to extract from summary.xlsx 'socket view' (defaults to a curated set)",
    )
    args = ap.parse_args()

    pd, plt = _try_import()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        _eprint(f"ERROR: run_dir not found or not a directory: {run_dir}")
        return 2

    agg = run_dir / "aggregate.csv"
    if not agg.exists():
        _eprint(f"ERROR: aggregate.csv not found: {agg}")
        return 2

    out_dir = run_dir / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean up legacy plot names from older versions of this script.
    for old in ["plot_tps_vs_token_len.png", "plot_tps_per_watt_vs_token_len.png"]:
        try:
            (out_dir / old).unlink()
        except FileNotFoundError:
            pass

    # Clean up deprecated output.
    try:
        (out_dir / "emon_metrics.csv").unlink()
    except FileNotFoundError:
        pass

    df = pd.read_csv(agg)
    for c in ["variant", "job_name", "resource_cpu", "sglang_max_total_tokens"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # Identify successful job rows (exclude marker rows created for variant failures).
    if "exit_code" in df.columns:
        df["exit_code"] = pd.to_numeric(df["exit_code"], errors="coerce")
    else:
        df["exit_code"] = math.nan

    is_job = df.get("job_name", "").astype(str).str.len() > 0
    is_ok = (df["exit_code"] == 0) if "exit_code" in df.columns else True
    df_jobs = df[is_job & is_ok].copy()

    # Coerce numeric columns.
    for c in ["batch_size", "token_len"]:
        if c in df_jobs.columns:
            df_jobs[c] = pd.to_numeric(df_jobs[c], errors="coerce")
    for c in ["tps", "latency_sec", "tps_per_cpu", "avg_batch_time_sec"]:
        if c in df_jobs.columns:
            df_jobs[c] = pd.to_numeric(df_jobs[c], errors="coerce")

    # Derived fields for scalability analysis.
    if "resource_cpu_count" in df_jobs.columns:
        df_jobs["cpu_count"] = pd.to_numeric(df_jobs["resource_cpu_count"], errors="coerce")
    else:
        df_jobs["cpu_count"] = math.nan

    if "sglang_max_total_tokens" in df_jobs.columns:
        df_jobs["kv_cap"] = pd.to_numeric(df_jobs["sglang_max_total_tokens"], errors="coerce")
    else:
        df_jobs["kv_cap"] = math.nan

    if all(c in df_jobs.columns for c in ["tps", "token_len"]):
        df_jobs["tokens_per_sec"] = df_jobs["tps"] * df_jobs["token_len"]
    else:
        df_jobs["tokens_per_sec"] = math.nan

    if all(c in df_jobs.columns for c in ["tokens_per_sec", "cpu_count"]):
        df_jobs["tokens_per_sec_per_cpu"] = df_jobs.apply(
            lambda r: (r["tokens_per_sec"] / r["cpu_count"]) if (pd.notna(r["cpu_count"]) and r["cpu_count"] > 0) else math.nan,
            axis=1,
        )
    else:
        df_jobs["tokens_per_sec_per_cpu"] = math.nan

    # ------------------------------------------------------------------
    # 1) failed_variants.csv
    # ------------------------------------------------------------------
    fail_rows = df[df["exit_code"].fillna(0) != 0].copy()
    failed_variants = (
        fail_rows[["variant", "resource_cpu", "resource_cpu_count", "sglang_max_total_tokens", "exit_code"]]
        .fillna("")
        .drop_duplicates()
    )

    # Add a small summary per variant.
    summary = (
        df.groupby("variant", as_index=False)
        .agg(
            num_rows=("variant", "size"),
            num_success_jobs=("exit_code", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) == 0).sum())),
            num_failed_rows=("exit_code", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) != 0).sum())),
        )
        .fillna("")
    )

    if not failed_variants.empty:
        failed_variants = failed_variants.merge(summary, on="variant", how="left")
        # Attach likely log path.
        failed_variants["auto_test_stdout_log"] = failed_variants["variant"].apply(
            lambda v: _guess_auto_test_stdout_log(run_dir, str(v))
        )

    _write_csv(failed_variants, out_dir / "failed_variants.csv")

    # ------------------------------------------------------------------
    # 2) summary_pivot.csv
    # ------------------------------------------------------------------
    # Group duplicates (e.g. repeats) by mean.
    group_cols = [c for c in ["variant", "resource_cpu", "resource_cpu_count", "resource_mem_gb", "sglang_max_total_tokens", "batch_size", "token_len"] if c in df_jobs.columns]
    metric_cols = [c for c in ["tps", "latency_sec", "tps_per_cpu"] if c in df_jobs.columns]

    if group_cols and metric_cols and not df_jobs.empty:
        # Some optional dimensions (e.g. resource_mem_gb) may be empty/NaN for all rows;
        # pandas groupby drops NaN groups by default, which would incorrectly erase the run.
        df_g = df_jobs.groupby(group_cols, as_index=False, dropna=False)[metric_cols].mean(numeric_only=True)

        # Some index dimensions may be entirely missing (NaN). If any index
        # column is NaN, pivot_table can drop the row; normalize missing values
        # to empty strings for stability.
        if "resource_mem_gb" in df_g.columns:
            df_g["resource_mem_gb"] = df_g["resource_mem_gb"].fillna("")

        # Make wide pivot for TPS and latency.
        index_cols = [c for c in ["variant", "resource_cpu", "resource_cpu_count", "resource_mem_gb", "sglang_max_total_tokens"] if c in df_g.columns]

        def _pivot(value_col: str, suffix: str) -> Any:
            pv = df_g.pivot_table(
                index=index_cols,
                columns=["token_len", "batch_size"],
                values=value_col,
                aggfunc="mean",
            )
            pv = pv.reset_index()
            # Flatten columns: token_len/batch_size combos become columns.
            # IMPORTANT: explicitly set meta column names to index_cols (strings),
            # because pandas may produce tuple-like labels for the reset index.
            old_cols = list(pv.columns)
            combo_cols = old_cols[len(index_cols) :]
            pv.columns = list(index_cols) + _flatten_pivot_columns(combo_cols, suffix)
            return pv

        pivots = [
            _pivot("tps", "tps") if "tps" in df_g.columns else None,
            _pivot("latency_sec", "latency_sec") if "latency_sec" in df_g.columns else None,
            _pivot("tps_per_cpu", "tps_per_cpu") if "tps_per_cpu" in df_g.columns else None,
        ]

        base = next((p for p in pivots if p is not None), None)
        if base is None:
            pd.DataFrame([]).to_csv(out_dir / "summary_pivot.csv", index=False)
        else:
            out = base.set_index(index_cols)
            for extra in pivots:
                if extra is None or extra is base:
                    continue
                out = out.join(extra.set_index(index_cols), how="outer")
            out = out.reset_index()
            _write_csv(out, out_dir / "summary_pivot.csv")
    else:
        # Still write an empty file for automation stability.
        pd.DataFrame([]).to_csv(out_dir / "summary_pivot.csv", index=False)

    # ------------------------------------------------------------------
    # 2b) Scalability CSVs
    # ------------------------------------------------------------------
    # We analyze mean behavior per unique combination; this also collapses repeats.
    base_dims = [c for c in ["resource_cpu", "cpu_count", "kv_cap", "sglang_max_total_tokens", "batch_size", "token_len"] if c in df_jobs.columns]
    value_dims = [c for c in ["tps", "tps_per_cpu", "tokens_per_sec", "tokens_per_sec_per_cpu", "avg_batch_time_sec"] if c in df_jobs.columns]
    # NOTE: kv_cap is numeric-parsed from sglang_max_total_tokens. When KV cap is
    # "auto" it becomes NaN; pandas groupby drops NaN groups by default, which
    # would incorrectly produce empty scalability CSVs. Use dropna=False so
    # "auto" runs still appear in scaling outputs.
    df_mean = (
        df_jobs.groupby(base_dims, as_index=False, dropna=False)[value_dims]
        .mean(numeric_only=True)
        .copy()
        if (base_dims and value_dims and not df_jobs.empty)
        else pd.DataFrame([])
    )

    def _sort_by(df_in: Any, cols: List[str]) -> Any:
        cols2 = [c for c in cols if c in df_in.columns]
        return df_in.sort_values(cols2, kind="mergesort") if cols2 else df_in

    # token_len scalability: keep (cpu_count, kv_cap, batch_size) fixed, vary token_len.
    if not df_mean.empty:
        token_len_scaling = _sort_by(df_mean, ["resource_cpu", "kv_cap", "batch_size", "token_len"])
        _write_csv(token_len_scaling, out_dir / "token_len_scaling.csv")

        batch_size_scaling = _sort_by(df_mean, ["resource_cpu", "kv_cap", "token_len", "batch_size"])
        _write_csv(batch_size_scaling, out_dir / "batch_size_scaling.csv")

        cpu_scaling = _sort_by(df_mean, ["kv_cap", "token_len", "batch_size", "cpu_count"])
        _write_csv(cpu_scaling, out_dir / "cpu_scaling.csv")

        kv_scaling = _sort_by(df_mean, ["cpu_count", "token_len", "batch_size", "kv_cap"])
        _write_csv(kv_scaling, out_dir / "kv_scaling.csv")
    else:
        pd.DataFrame([]).to_csv(out_dir / "token_len_scaling.csv", index=False)
        pd.DataFrame([]).to_csv(out_dir / "batch_size_scaling.csv", index=False)
        pd.DataFrame([]).to_csv(out_dir / "cpu_scaling.csv", index=False)
        pd.DataFrame([]).to_csv(out_dir / "kv_scaling.csv", index=False)

    # ------------------------------------------------------------------
    # 3) emon_socket_metrics.csv (socket view)
    # ------------------------------------------------------------------
    socket_metrics = list(args.socket_metrics) if args.socket_metrics else list(DEFAULT_SOCKET_VIEW_METRICS)
    socket_rows: List[Dict[str, Any]] = []

    if "emon_summary_xlsx" in df_jobs.columns and not df_jobs.empty:
        cache: Dict[str, Dict[str, Any]] = {}
        for _, r in df_jobs.iterrows():
            xlsx_raw = r.get("emon_summary_xlsx")
            try:
                if pd.isna(xlsx_raw):
                    continue
            except Exception:
                pass
            xlsx = str(xlsx_raw or "").strip()
            if not xlsx or xlsx.lower() in {"nan", "none"}:
                continue
            try:
                if not Path(xlsx).is_file():
                    continue
            except Exception:
                continue
            if xlsx not in cache:
                cache[xlsx] = _extract_socket_view_from_xlsx(pd=pd, xlsx_path=xlsx, wanted_metrics=socket_metrics)
            rec: Dict[str, Any] = {
                "server_host": r.get("server_host", ""),
                "variant": r.get("variant", ""),
                "job_name": r.get("job_name", ""),
                "resource_cpu": r.get("resource_cpu", ""),
                "resource_cpu_count": r.get("resource_cpu_count", ""),
                "sglang_max_total_tokens": r.get("sglang_max_total_tokens", ""),
                "batch_size": r.get("batch_size", ""),
                "token_len": r.get("token_len", ""),
                "emon_summary_xlsx": xlsx,
            }
            rec.update(cache.get(xlsx, {}))
            socket_rows.append(rec)

    if socket_rows:
        df_socket = pd.DataFrame(socket_rows)
        _write_csv(df_socket, out_dir / "emon_socket_metrics.csv")
    else:
        pd.DataFrame([]).to_csv(out_dir / "emon_socket_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 3c) EMON socket TMA pie charts + per-run HTML summary
    # ------------------------------------------------------------------
    # Pie chart: for each socket, show 4 top-level TMA percentages.
    tma_top = [
        "metric_TMA_Frontend_Bound(%)",
        "metric_TMA_Bad_Speculation(%)",
        "metric_TMA_Retiring(%)",
        "metric_TMA_Backend_Bound(%)",
    ]

    df_socket_loaded = None
    try:
        df_socket_loaded = pd.read_csv(out_dir / "emon_socket_metrics.csv")
    except Exception:
        df_socket_loaded = None

    if df_socket_loaded is not None and not df_socket_loaded.empty and plt is not None:
        # --------------------------------------------------------------
        # Per-job socket TMA pie charts (job page in web UI)
        # --------------------------------------------------------------
        # We write a manifest so the web server can locate images
        # without guessing filename sanitization rules.
        job_pies_dir = out_dir / "emon_job"
        manifest: Dict[str, Any] = {}
        try:
            socket_cols2 = [c for c in df_socket_loaded.columns if c.startswith("socket_") and "__" in c]
            sockets2 = sorted({c.split("__", 1)[0] for c in socket_cols2})
        except Exception:
            sockets2 = []

        for _, row in df_socket_loaded.iterrows():
            server_host = str(row.get("server_host", "") or "").strip()
            job_name = str(row.get("job_name", "") or "").strip()
            if not job_name:
                continue
            jid = _sanitize_job_id(f"{server_host}__{job_name}" if server_host else job_name)
            rec: Dict[str, Any] = {"server_host": server_host, "job_name": job_name, "job_id": jid, "pies": {}}
            out_job_dir = job_pies_dir / jid
            out_job_dir.mkdir(parents=True, exist_ok=True)

            for s in sockets2:
                vals: List[float] = []
                labels: List[str] = []
                for m in tma_top:
                    col = f"{s}__{m}"
                    if col not in df_socket_loaded.columns:
                        continue
                    try:
                        v = pd.to_numeric(row.get(col), errors="coerce")
                        try:
                            if pd.isna(v):
                                continue
                        except Exception:
                            if v != v:
                                continue
                        vals.append(float(v))
                        labels.append(m.replace("metric_TMA_", "").replace("(%)", ""))
                    except Exception:
                        continue
                if not vals:
                    continue
                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                ax.pie(
                    vals,
                    labels=labels,
                    autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
                    startangle=90,
                    counterclock=False,
                )
                ax.set_title(f"{job_name} | {s} TMA top-level (%)")
                fig.tight_layout()
                out_png = out_job_dir / f"tma_pie_{s}.png"
                fig.savefig(out_png, dpi=160)
                plt.close(fig)
                rel_png = out_png.relative_to(run_dir).as_posix()
                rec["pies"][s] = rel_png

            if rec.get("pies"):
                key = f"{server_host}::{job_name}" if server_host else job_name
                manifest[key] = rec

        try:
            _write_text(out_dir / "emon_job_pies_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        except Exception:
            pass

        socket_cols = [c for c in df_socket_loaded.columns if c.startswith("socket_") and "__" in c]
        sockets = sorted({c.split("__", 1)[0] for c in socket_cols})
        for s in sockets:
            vals: List[float] = []
            labels: List[str] = []
            for m in tma_top:
                col = f"{s}__{m}"
                if col not in df_socket_loaded.columns:
                    continue
                ser = pd.to_numeric(df_socket_loaded[col], errors="coerce").dropna()
                if ser.empty:
                    continue
                labels.append(m.replace("metric_TMA_", "").replace("(%)", ""))
                vals.append(float(ser.mean()))
            if not vals:
                continue
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            ax.pie(
                vals,
                labels=labels,
                autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
                startangle=90,
                counterclock=False,
            )
            ax.set_title(f"{s} TMA top-level (%)")
            fig.tight_layout()
            fig.savefig(out_dir / f"emon_socket_tma_pie_{s}.png", dpi=160)
            plt.close(fig)

    # Per-run summary HTML (rendered by the web UI)
    run_summary_metrics = list(DEFAULT_SOCKET_VIEW_METRICS)
    try:
        df_socket_for_summary = df_socket_loaded if df_socket_loaded is not None else pd.DataFrame([])
    except Exception:
        df_socket_for_summary = pd.DataFrame([])

    summary_html = _render_run_summary_html(
        run_dir=run_dir,
        out_dir=out_dir,
        df_jobs=df_jobs,
        failed_variants=failed_variants,
        df_socket=df_socket_for_summary,
        wanted_metrics=run_summary_metrics,
    )
    _write_text(out_dir / "run_summary.html", summary_html)

    # ------------------------------------------------------------------
    # 4) Plots (matplotlib)
    # ------------------------------------------------------------------
    # Scalability plots: focus on 4 dimensions.
    # We use df_mean (mean across repeats) to keep plots readable.
    if plt is not None and (not df_mean.empty):
        # Helper: plot lines by a categorical variable.
        def _plot_lines(
            *,
            ax: Any,
            data: Any,
            x: str,
            y: str,
            line_by: str,
            title: str,
            xlabel: str,
            ylabel: str,
            annotate_points: bool = True,
        ) -> None:
            if data.empty:
                ax.set_axis_off()
                return

            g = data.dropna(subset=[x, y]).copy()
            if g.empty:
                ax.set_axis_off()
                return

            for line_idx, (k, g2) in enumerate(g.groupby(line_by)):
                g2 = g2.sort_values(x)
                ax.plot(g2[x], g2[y], marker="o", linewidth=1.8, label=f"{line_by}={k}")
                if annotate_points:
                    xs = list(g2[x])
                    ys = list(g2[y])
                    for px, py in zip(xs, ys):
                        s = _fmt_point_value(py)
                        if not s:
                            continue
                        ax.annotate(
                            s,
                            (px, py),
                            textcoords="offset points",
                            xytext=(0, 6 + 8 * (line_idx % 3)),
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            alpha=0.9,
                        )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

        cpu_vals = sorted(set(df_mean["resource_cpu"].astype(str)), key=_safe_int_sort_key) if "resource_cpu" in df_mean.columns else []
        kv_vals = sorted(set(df_mean["kv_cap"].dropna().astype(int)), key=lambda x: x) if "kv_cap" in df_mean.columns else []
        tl_vals = sorted(set(df_mean["token_len"].dropna().astype(int)), key=lambda x: x) if "token_len" in df_mean.columns else []

        # Plot A: token_len scalability (use tokens_per_sec to normalize by input size)
        if all(c in df_mean.columns for c in ["resource_cpu", "kv_cap", "batch_size", "token_len", "tokens_per_sec"]):
            nrows = max(1, len(cpu_vals))
            ncols = max(1, len(kv_vals))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
            for i, cpu in enumerate(cpu_vals):
                for j, kv in enumerate(kv_vals):
                    ax = axes[i][j]
                    sub = df_mean[(df_mean["resource_cpu"].astype(str) == cpu) & (df_mean["kv_cap"].astype(float) == float(kv))].copy()
                    _plot_lines(
                        ax=ax,
                        data=sub,
                        x="token_len",
                        y="tokens_per_sec",
                        line_by="batch_size",
                        title=f"Token-len scaling | cpu={cpu} | kv={kv}",
                        xlabel="token_len",
                        ylabel="tokens/sec (tps * token_len)",
                    )
            fig.tight_layout()
            fig.savefig(out_dir / "plot_token_len_scaling.png", dpi=160)
            plt.close(fig)

        # Plot B: batch_size scalability (TPS vs batch_size)
        if all(c in df_mean.columns for c in ["resource_cpu", "kv_cap", "batch_size", "token_len", "tps"]):
            nrows = max(1, len(cpu_vals))
            ncols = max(1, len(kv_vals))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
            for i, cpu in enumerate(cpu_vals):
                for j, kv in enumerate(kv_vals):
                    ax = axes[i][j]
                    sub = df_mean[(df_mean["resource_cpu"].astype(str) == cpu) & (df_mean["kv_cap"].astype(float) == float(kv))].copy()
                    _plot_lines(
                        ax=ax,
                        data=sub,
                        x="batch_size",
                        y="tps",
                        line_by="token_len",
                        title=f"Batch scaling | cpu={cpu} | kv={kv}",
                        xlabel="batch_size",
                        ylabel="samples/sec (tps)",
                    )
            fig.tight_layout()
            fig.savefig(out_dir / "plot_batch_size_scaling.png", dpi=160)
            plt.close(fig)

        # Plot C: CPU scaling (TPS vs cpu_count)
        if all(c in df_mean.columns for c in ["cpu_count", "kv_cap", "batch_size", "token_len", "tps"]):
            # Make a grid by kv_cap (cols) and token_len (rows)
            nrows = max(1, len(tl_vals))
            ncols = max(1, len(kv_vals))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
            for i, tl in enumerate(tl_vals):
                for j, kv in enumerate(kv_vals):
                    ax = axes[i][j]
                    sub = df_mean[(df_mean["token_len"].astype(float) == float(tl)) & (df_mean["kv_cap"].astype(float) == float(kv))].copy()
                    # Use batch_size as lines (common scaling dimension)
                    _plot_lines(
                        ax=ax,
                        data=sub,
                        x="cpu_count",
                        y="tps",
                        line_by="batch_size",
                        title=f"CPU scaling | tok={tl} | kv={kv}",
                        xlabel="CPU cores (count)",
                        ylabel="samples/sec (tps)",
                    )
            fig.tight_layout()
            fig.savefig(out_dir / "plot_cpu_scaling.png", dpi=160)
            plt.close(fig)

        # Plot D: KV cap scaling (TPS vs kv_cap)
        if all(c in df_mean.columns for c in ["cpu_count", "kv_cap", "batch_size", "token_len", "tps"]):
            cpu_counts = sorted(set(df_mean["cpu_count"].dropna().astype(int)), key=lambda x: x)
            nrows = max(1, len(cpu_counts))
            ncols = max(1, len(tl_vals))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
            for i, cc in enumerate(cpu_counts):
                for j, tl in enumerate(tl_vals):
                    ax = axes[i][j]
                    sub = df_mean[(df_mean["cpu_count"].astype(float) == float(cc)) & (df_mean["token_len"].astype(float) == float(tl))].copy()
                    _plot_lines(
                        ax=ax,
                        data=sub,
                        x="kv_cap",
                        y="tps",
                        line_by="batch_size",
                        title=f"KV scaling | cpu={cc} | tok={tl}",
                        xlabel="SGLANG max_total_tokens (cap)",
                        ylabel="samples/sec (tps)",
                    )
            fig.tight_layout()
            fig.savefig(out_dir / "plot_kv_cap_scaling.png", dpi=160)
            plt.close(fig)


    print(f"[ok] Wrote: {out_dir / 'summary_pivot.csv'}")
    print(f"[ok] Wrote: {out_dir / 'emon_socket_metrics.csv'}")
    print(f"[ok] Wrote: {out_dir / 'failed_variants.csv'}")
    for p in [
        out_dir / "token_len_scaling.csv",
        out_dir / "batch_size_scaling.csv",
        out_dir / "cpu_scaling.csv",
        out_dir / "kv_scaling.csv",
    ]:
        if p.exists():
            print(f"[ok] Wrote: {p}")
    for p in [
        out_dir / "plot_token_len_scaling.png",
        out_dir / "plot_batch_size_scaling.png",
        out_dir / "plot_cpu_scaling.png",
        out_dir / "plot_kv_cap_scaling.png",
    ]:
        if p.exists():
            print(f"[ok] Plot:  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
