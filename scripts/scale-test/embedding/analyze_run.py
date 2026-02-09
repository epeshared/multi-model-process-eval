#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_METRICS = [
    "metric_CPU operating frequency (in GHz)",
    "metric_uncore frequency GHz",
    "metric_DDR data rate (MT/sec)",
    "metric_memory bandwidth total (MB/sec)",
    "metric_CPI",
    "metric_core IPC",
    "metric_CPU utilization %",
    "metric_package power (watts)",
    "metric_NUMA %_Reads addressed to remote DRAM",
]


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _try_import() -> Tuple[Any, Any]:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        _eprint("ERROR: pandas is required for analysis")
        _eprint(f"Import error: {e}")
        _eprint("Try: python -m pip install pandas")
        raise

    try:
        import matplotlib  # type: ignore

        # Headless-safe backend (common on servers / SSH).
        matplotlib.use("Agg", force=True)  # type: ignore[attr-defined]
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        _eprint("ERROR: matplotlib is required for plots")
        _eprint(f"Import error: {e}")
        _eprint("Try: python -m pip install matplotlib")
        raise

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
        help="EMON metric keys to extract (defaults to a curated set)",
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
            lambda v: str((run_dir / v / "auto_test_stdout.log").resolve())
        )

    _write_csv(failed_variants, out_dir / "failed_variants.csv")

    # ------------------------------------------------------------------
    # 2) summary_pivot.csv
    # ------------------------------------------------------------------
    # Group duplicates (e.g. repeats) by mean.
    group_cols = [c for c in ["variant", "resource_cpu", "resource_cpu_count", "resource_mem_gb", "sglang_max_total_tokens", "batch_size", "token_len"] if c in df_jobs.columns]
    metric_cols = [c for c in ["tps", "latency_sec", "tps_per_cpu"] if c in df_jobs.columns]

    if group_cols and metric_cols and not df_jobs.empty:
        df_g = df_jobs.groupby(group_cols, as_index=False)[metric_cols].mean(numeric_only=True)

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
    df_mean = (
        df_jobs.groupby(base_dims, as_index=False)[value_dims]
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
    # 3) emon_metrics.csv
    # ------------------------------------------------------------------
    metrics = list(args.metrics) if args.metrics else list(DEFAULT_METRICS)

    if "emon_kv_json" in df_jobs.columns:
        rows: List[Dict[str, Any]] = []
        for _, r in df_jobs.iterrows():
            emon_json = str(r.get("emon_kv_json") or "").strip()
            if not emon_json:
                continue
            try:
                kv = json.loads(emon_json)
            except Exception:
                continue

            rec: Dict[str, Any] = {
                "variant": r.get("variant", ""),
                "job_name": r.get("job_name", ""),
                "resource_cpu": r.get("resource_cpu", ""),
                "resource_cpu_count": r.get("resource_cpu_count", ""),
                "resource_mem_gb": r.get("resource_mem_gb", ""),
                "sglang_max_total_tokens": r.get("sglang_max_total_tokens", ""),
                "batch_size": r.get("batch_size", ""),
                "token_len": r.get("token_len", ""),
                "tps": r.get("tps", ""),
                "latency_sec": r.get("latency_sec", ""),
                "tps_per_cpu": r.get("tps_per_cpu", ""),
                "emon_summary_xlsx": r.get("emon_summary_xlsx", ""),
                "emon_output_path": r.get("emon_output_path", ""),
            }

            for k in metrics:
                rec[k] = kv.get(k, "")

            # Convenience: efficiency
            pw = _to_float(kv.get("metric_package power (watts)"))
            tps = _to_float(r.get("tps"))
            rec["tps_per_watt"] = (tps / pw) if (pw and pw > 0 and tps is not None) else ""

            rows.append(rec)

        df_emon = pd.DataFrame(rows)
        _write_csv(df_emon, out_dir / "emon_metrics.csv")
    else:
        pd.DataFrame([]).to_csv(out_dir / "emon_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 4) Plots (matplotlib)
    # ------------------------------------------------------------------
    # Scalability plots: focus on 4 dimensions.
    # We use df_mean (mean across repeats) to keep plots readable.
    if not df_mean.empty:
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
    print(f"[ok] Wrote: {out_dir / 'emon_metrics.csv'}")
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
