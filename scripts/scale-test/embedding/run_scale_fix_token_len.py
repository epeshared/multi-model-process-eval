#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_compact() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _which(bin_name: str) -> Optional[str]:
    try:
        return shutil.which(bin_name)
    except Exception:
        return None


def _parse_cpus(expr: str) -> List[int]:
    s = (expr or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo = int(a.strip())
                hi = int(b.strip())
            except Exception:
                continue
            if hi < lo:
                lo, hi = hi, lo
            for x in range(lo, hi + 1):
                out.append(x)
        else:
            try:
                out.append(int(part))
            except Exception:
                continue
    return sorted(set(out))


def _cpu_count(expr: str) -> int:
    """Return number of logical CPUs described by a linux CPU list expression."""

    try:
        return len(_parse_cpus(expr))
    except Exception:
        return 0


def _cpu_to_numa_node(cpu_id: int) -> Optional[int]:
    """Best-effort map CPU id -> NUMA node id using sysfs."""

    cpu_dir = Path(f"/sys/devices/system/cpu/cpu{int(cpu_id)}")
    if not cpu_dir.exists():
        return None
    try:
        for child in cpu_dir.iterdir():
            name = child.name
            if name.startswith("node") and name[4:].isdigit():
                return int(name[4:])
    except Exception:
        return None
    return None


def _infer_single_numa_node_from_cpu_expr(cpu_expr: str) -> Optional[int]:
    """Infer a single NUMA node id if all requested CPUs map to the same node."""

    cpus = _parse_cpus(cpu_expr)
    if not cpus:
        return None

    nodes: List[int] = []
    for cpu in cpus[:256]:
        n = _cpu_to_numa_node(cpu)
        if n is None:
            return None
        nodes.append(n)

    uniq = sorted(set(nodes))
    if len(uniq) == 1:
        return uniq[0]
    return None


def _bytes_from_gb(gb: Optional[float]) -> Optional[int]:
    if gb is None:
        return None
    try:
        return int(float(gb) * (1024**3))
    except Exception:
        return None


def _try_systemd_scope_cmd(
    *,
    cmd: List[str],
    cpu_expr: str,
    mem_bytes: Optional[int],
) -> Optional[List[str]]:
    systemd_run = _which("systemd-run")
    if not systemd_run:
        return None

    props: List[str] = []
    if (cpu_expr or "").strip():
        # For transient scope units, systemd commonly supports AllowedCPUs= (cpuset).
        # CPUAffinity= is a service property and may be rejected for scopes.
        props += ["-p", f"AllowedCPUs={cpu_expr.strip()}"]
    if mem_bytes and mem_bytes > 0:
        props += ["-p", f"MemoryMax={mem_bytes}"]

    # Try user scope first (works in most non-root environments).
    try:
        test = subprocess.run(
            [systemd_run, "--user", "--scope", "--quiet", "--", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if test.returncode == 0:
            return [systemd_run, "--user", "--scope"] + props + ["--"] + cmd
    except Exception:
        pass

    # Fallback: system scope (may require privileges but sometimes works).
    try:
        test = subprocess.run(
            [systemd_run, "--scope", "--quiet", "--", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if test.returncode == 0:
            return [systemd_run, "--scope"] + props + ["--"] + cmd
    except Exception:
        pass

    return None


def _wrap_cmd_taskset_prlimit(
    *,
    cmd: List[str],
    cpu_expr: str,
    mem_bytes: Optional[int],
) -> List[str]:
    out = list(cmd)

    prlimit = _which("prlimit")
    if mem_bytes and mem_bytes > 0 and prlimit:
        out = [prlimit, f"--as={mem_bytes}", "--"] + out

    taskset = _which("taskset")
    if (cpu_expr or "").strip() and taskset:
        out = [taskset, "-c", cpu_expr.strip()] + out

    return out


def _constrained_cmd(*, cmd: List[str], cpu_expr: str, mem_bytes: Optional[int]) -> Tuple[List[str], str]:
    # If no constraints are requested, return the command unchanged.
    if not (cpu_expr or "").strip() and not (mem_bytes and mem_bytes > 0):
        return list(cmd), "none"

    sysd = _try_systemd_scope_cmd(cmd=cmd, cpu_expr=cpu_expr, mem_bytes=mem_bytes)
    if sysd is not None:
        return sysd, "systemd-run"

    wrapped = _wrap_cmd_taskset_prlimit(cmd=cmd, cpu_expr=cpu_expr, mem_bytes=mem_bytes)
    method = "taskset/prlimit" if wrapped != cmd else "none"
    return wrapped, method


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (s or "").strip())


def _as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


@dataclass(frozen=True)
class ScaleConfig:
    template_auto_test_config: Path
    result_root: Path
    token_lens: List[int]
    batch_sizes: List[int]
    repeats: int
    warmup_runs: int
    continue_on_error: bool
    cpu_exprs: List[str]
    mem_gb: Optional[float]
    sglang_max_total_tokens: List[str]
    job_template: Dict[str, Any]
    emon_process_after_run: bool
    emon_process_cmd: List[str]
    emon_expected_output: str


def _parse_scale_config(cfg_path: Path) -> ScaleConfig:
    raw = _load_json(cfg_path)
    template = Path(str(raw.get("template_auto_test_config") or "scripts/auto-test/embedding/config_fix_token_len.json"))
    if not template.is_absolute():
        template = (REPO_ROOT / template).resolve()

    result_root = Path(str(raw.get("result_root") or "scripts/scale-test/embedding/result/fix_token_len"))
    if not result_root.is_absolute():
        result_root = (REPO_ROOT / result_root).resolve()

    run = raw.get("run") or {}
    token_lens = [int(x) for x in (run.get("token_lens") or [])]
    if not token_lens:
        raise SystemExit("config.run.token_lens is empty")

    batch_sizes_raw = run.get("batch_sizes")
    batch_sizes: List[int] = []
    if batch_sizes_raw is not None:
        batch_sizes = [int(x) for x in (batch_sizes_raw or [])]

    repeats = int(run.get("repeats") or 1)
    warmup_runs = int(run.get("warmup_runs") or 0)

    continue_on_error = bool(run.get("continue_on_error") or run.get("continue_on_failure") or False)

    cpu_raw = (run.get("cpu") or {}).get("cpus")
    cpu_exprs = _as_str_list(cpu_raw)
    if not cpu_exprs:
        cpu_exprs = [""]

    mem_gb = (run.get("memory") or {}).get("max_gb")
    mem_gb_f: Optional[float] = None
    if mem_gb is not None and str(mem_gb).strip() != "":
        mem_gb_f = float(mem_gb)

    sglang_mtt = _as_str_list(run.get("sglang_max_total_tokens"))

    job_template = raw.get("job_template") or {}
    if not isinstance(job_template, dict):
        raise SystemExit("config.job_template must be an object")

    # Backward compatible: if run.sglang_max_total_tokens is not provided,
    # we still allow a single value in job_template.extra_env.
    if not sglang_mtt:
        extra_env = job_template.get("extra_env") or {}
        if not isinstance(extra_env, dict):
            extra_env = {}
        v = str(extra_env.get("SGLANG_MAX_TOTAL_TOKENS") or "").strip()
        sglang_mtt = [v] if v else [""]

    if not batch_sizes:
        # Default to job_template.batch_size if not specified.
        try:
            batch_sizes = [int((job_template.get("batch_size") or 0))]
        except Exception:
            batch_sizes = []
    if not batch_sizes or any(bs <= 0 for bs in batch_sizes):
        raise SystemExit("config.run.batch_sizes is empty/invalid (or job_template.batch_size invalid)")

    emon = raw.get("emon") or {}
    emon_process_after_run = bool(emon.get("process_after_run", True))
    process_cmd = emon.get("process_cmd") or [
        "emon",
        "-process-pyedp",
        "/opt/intel/sep/config/edp/pyedp_config.txt",
    ]
    if not isinstance(process_cmd, list) or not all(isinstance(x, str) for x in process_cmd):
        raise SystemExit("config.emon.process_cmd must be a string list")
    expected_output = str(emon.get("expected_output") or "summary.xlsx")

    return ScaleConfig(
        template_auto_test_config=template,
        result_root=result_root,
        token_lens=token_lens,
        batch_sizes=batch_sizes,
        repeats=repeats,
        warmup_runs=warmup_runs,
        continue_on_error=continue_on_error,
        cpu_exprs=cpu_exprs,
        mem_gb=mem_gb_f,
        sglang_max_total_tokens=sglang_mtt,
        job_template=job_template,
        emon_process_after_run=emon_process_after_run,
        emon_process_cmd=[str(x) for x in process_cmd],
        emon_expected_output=expected_output,
    )


def _generate_auto_test_config(
    *,
    scale: ScaleConfig,
    result_dir: Path,
    cpu_expr: str,
    sglang_max_total_tokens: str,
    variant_tag: str = "",
) -> Dict[str, Any]:
    template = _load_json(scale.template_auto_test_config)
    if not isinstance(template, dict):
        raise SystemExit(f"template auto-test config is not an object: {scale.template_auto_test_config}")

    cfg = dict(template)
    defaults = dict(cfg.get("defaults") or {})
    defaults["result_dir"] = str(result_dir)
    # Let the scale-test control warmup via per-job warmup_runs.
    cfg["defaults"] = defaults

    jt = scale.job_template
    backend = str(jt.get("backend") or "sglang").strip()
    model = str(jt.get("model") or "").strip()
    if not model:
        raise SystemExit("job_template.model is required")
    model_dir = str(jt.get("model_dir") or "").strip()
    base_url = str(jt.get("base_url") or "").strip()

    host = str(jt.get("host") or "0.0.0.0").strip()
    port = int(jt.get("port") or 30000)
    served_model_name = str(jt.get("served_model_name") or jt.get("model_id") or "").strip()
    model_id = str(jt.get("model_id") or served_model_name or model).strip()

    mode = str(jt.get("mode") or "token_len").strip()
    max_samples = int(jt.get("max_samples") or 1000)
    batch_size_default = int(jt.get("batch_size") or 100)
    dtype = str(jt.get("dtype") or "bfloat16").strip()

    restart_servers = bool(jt.get("restart_servers", True))
    stop_server_after_job = bool(jt.get("stop_server_after_job", True))
    emon_enable = bool(jt.get("emon_enable", False))
    extra_env = jt.get("extra_env") or {}
    if not isinstance(extra_env, dict):
        extra_env = {}

    # -------------------------
    # Auto NUMA binding (best-effort)
    #
    # Goal: if the user constrains CPUs (e.g. 0-31), we can often infer the
    # NUMA node and bind server + benchmark to local memory for better
    # locality and more predictable behavior under MemoryMax.
    #
    # We only apply this when:
    # - scale.cpu_expr is set
    # - user did not explicitly specify NUMACTL_* / SERVER_NUMACTL_* in extra_env
    # - server spec does not already configure servers.<backend>.numactl
    #
    # If CPUs span multiple nodes, we still bind cores but skip membind/cpunodebind.
    # -------------------------
    inferred_node: Optional[int] = None
    if (cpu_expr or "").strip():
        inferred_node = _infer_single_numa_node_from_cpu_expr(cpu_expr)

    # If the chosen backend uses an HTTP server managed by auto-test, we may
    # need to forward extra env vars into the server start script.
    # In particular, the sglang server script supports SGLANG_PYTHON to pick
    # the correct interpreter (important when running under systemd-run).
    try:
        servers = cfg.get("servers") or {}
        if isinstance(servers, dict) and backend in servers and isinstance(servers.get(backend), dict):
            s = dict(servers[backend])
            env_from_job = dict(s.get("env_from_job") or {})
            for k in extra_env.keys():
                ks = str(k)
                if ks.startswith("SGLANG_"):
                    env_from_job[ks] = ks
            # Also forward sweep-provided SGLANG settings that may not exist in extra_env.
            if str(sglang_max_total_tokens or "").strip():
                env_from_job["SGLANG_MAX_TOTAL_TOKENS"] = "SGLANG_MAX_TOTAL_TOKENS"
            s["env_from_job"] = env_from_job

            # Auto-populate server NUMA binding from scale.cpu_expr when safe.
            # (Server numactl is configured in the server spec, not via env vars.)
            user_server_override = any(
                str(extra_env.get(k) or "").strip()
                for k in ["SERVER_NUMACTL_CORES", "SERVER_NUMACTL_CPUNODEBIND", "SERVER_NUMACTL_MEMBIND"]
            )
            user_job_override = any(
                str(extra_env.get(k) or "").strip() for k in ["NUMACTL_CORES", "NUMACTL_CPUNODEBIND", "NUMACTL_MEMBIND"]
            )

            # Only touch server numactl if user didn't opt into explicit controls.
            # IMPORTANT: the auto-test template config may pin the server to a fixed core range
            # (e.g. 0-15). For scale-test we want the server to follow the scale CPU constraint.
            if (cpu_expr or "").strip() and not user_server_override and not user_job_override:
                numactl = s.get("numactl")
                numactl_obj: Dict[str, Any] = dict(numactl) if isinstance(numactl, dict) else {}

                # Always bind server cores to the scale CPU expr.
                numactl_obj["cores"] = cpu_expr.strip()

                # Bind memory locally when we can infer a single NUMA node.
                if inferred_node is not None and inferred_node >= 0:
                    numactl_obj["cpunodebind"] = str(inferred_node)
                    numactl_obj["membind"] = str(inferred_node)

                s["numactl"] = numactl_obj

            servers[backend] = s
            cfg["servers"] = servers
    except Exception:
        pass

    jobs: List[Dict[str, Any]] = []
    for rep in range(1, max(1, int(scale.repeats)) + 1):
        for bs in scale.batch_sizes:
            for tl in scale.token_lens:
                name = f"scale_fix_token_len_tok{int(tl)}_bs{int(bs)}_{model}_{backend}_rep{rep}"
                if (variant_tag or "").strip():
                    name = f"{name}__{_safe_name(variant_tag)}"
                env = {
                    "MODEL": model,
                    "BACKEND": backend,
                    "MODEL_DIR": model_dir,
                    "SERVED_MODEL_NAME": served_model_name,
                    "MODEL_ID": model_id,
                    "HOST": host,
                    "PORT": str(port),
                    "BASE_URL": base_url,
                    "MODE": mode,
                    "MAX_SAMPLES": str(max_samples),
                    "BATCH_SIZE": str(int(bs) if int(bs) > 0 else batch_size_default),
                    "SYNTHETIC_TOKEN_LEN": str(int(tl)),
                    "DTYPE": dtype,
                }

                # Merge user-provided env first (can override any defaults).
                for k, v in extra_env.items():
                    env[str(k)] = str(v)

                # Optional sweep override (server + job): SGLANG_MAX_TOTAL_TOKENS.
                if str(sglang_max_total_tokens or "").strip():
                    env["SGLANG_MAX_TOTAL_TOKENS"] = str(sglang_max_total_tokens).strip()

                # Auto-populate benchmark NUMA binding when safe.
                # (This wraps the job command with numactl in run_auto_test.py.)
                if (cpu_expr or "").strip() and not any(
                    str(extra_env.get(k) or "").strip()
                    for k in [
                        "NUMACTL_CORES",
                        "NUMACTL_CPUNODEBIND",
                        "NUMACTL_MEMBIND",
                        "SERVER_NUMACTL_CORES",
                        "SERVER_NUMACTL_CPUNODEBIND",
                        "SERVER_NUMACTL_MEMBIND",
                    ]
                ):
                    if not str(env.get("NUMACTL_CORES") or "").strip():
                        env["NUMACTL_CORES"] = cpu_expr.strip()
                    if inferred_node is not None and inferred_node >= 0:
                        if not str(env.get("NUMACTL_CPUNODEBIND") or "").strip():
                            env["NUMACTL_CPUNODEBIND"] = str(inferred_node)
                        if not str(env.get("NUMACTL_MEMBIND") or "").strip():
                            env["NUMACTL_MEMBIND"] = str(inferred_node)

                jobs.append(
                    {
                        "name": name,
                        "script": "run_fix_token_len",
                        "args": {},
                        "warmup_runs": int(scale.warmup_runs),
                        "restart_servers": restart_servers,
                        "stop_server_after_job": stop_server_after_job,
                        "emon_enable": emon_enable,
                        "env": env,
                    }
                )

    cfg["jobs"] = jobs
    return cfg


def _run_auto_test(
    *,
    auto_test_config_path: Path,
    work_dir: Path,
    tee: bool,
    dry_run: bool,
    cpu_expr: str,
    mem_gb: Optional[float],
    stdout_log: Path,
) -> int:
    runner = (REPO_ROOT / "scripts/auto-test/embedding/run_auto_test.py").resolve()
    if not runner.exists():
        raise SystemExit(f"Missing runner: {runner}")

    cmd = [sys.executable, str(runner), "--config", str(auto_test_config_path)]
    if tee:
        cmd.append("--tee")
    if dry_run:
        cmd.append("--dry-run")

    mem_bytes = _bytes_from_gb(mem_gb)
    full_cmd, method = _constrained_cmd(cmd=cmd, cpu_expr=cpu_expr, mem_bytes=mem_bytes)
    if method == "none" and ((cpu_expr or "").strip() or (mem_bytes and mem_bytes > 0)):
        print("[warn] No resource limiter found (systemd-run/taskset/prlimit). Running without enforcement.")
    else:
        print(f"[info] Resource limiting method: {method}")
        if (cpu_expr or "").strip():
            print(f"[info] CPU cores: {cpu_expr}")
        if mem_bytes and mem_bytes > 0:
            print(f"[info] MemoryMax(bytes): {mem_bytes}")

    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(
            full_cmd,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            if tee:
                try:
                    sys.stdout.write(line)
                except BrokenPipeError:
                    # If output is piped (e.g. `| head`) and the pipe closes early,
                    # keep running but stop teeing to stdout.
                    tee = False
        return int(p.wait())


def _find_single_run_id(result_dir: Path) -> str:
    # run_auto_test creates a subdir named run_id inside result_dir.
    candidates = [p.name for p in result_dir.iterdir() if p.is_dir()]
    # Filter to compact UTC IDs: 20260101T010203Z
    candidates = [c for c in candidates if len(c) == 16 and c.endswith("Z") and "T" in c]
    if not candidates:
        raise SystemExit(f"No run_id directories found under {result_dir}")
    return sorted(candidates)[-1]


def _iter_summary_rows(summary_csv: Path) -> Iterable[Dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k: (v if v is not None else "") for k, v in row.items()}


def _process_emon_dirs(*, rows: List[Dict[str, Any]], tee: bool, process_cmd: List[str], expected_output: str) -> None:
    emon_bin = _which("emon")
    if not emon_bin:
        print("[warn] emon not found in PATH; skipping emon -process-pyedp")
        return
    if not process_cmd:
        return

    for rec in rows:
        emon_out = str(rec.get("emon_output_path") or "").strip()
        if not emon_out:
            continue
        emon_dat = Path(emon_out)
        if not emon_dat.exists():
            continue
        emon_dir = emon_dat.parent
        out_xlsx = emon_dir / expected_output
        if out_xlsx.exists():
            rec["emon_summary_xlsx"] = str(out_xlsx)
            continue

        cmd = [process_cmd[0]] + process_cmd[1:]
        # Ensure we invoke the resolved emon binary if user used just "emon".
        if cmd and cmd[0] == "emon":
            cmd[0] = emon_bin
        if tee:
            print(f"[emon] processing in {emon_dir}: {' '.join(cmd)}")
        try:
            p = subprocess.run(cmd, cwd=str(emon_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            rec["emon_process_rc"] = int(p.returncode)
            rec["emon_process_log"] = str((emon_dir / "emon_process.log").resolve())
            (emon_dir / "emon_process.log").write_text(p.stdout or "", encoding="utf-8")
        except Exception as e:
            rec["emon_process_rc"] = 999
            rec["emon_process_err"] = str(e)

        if out_xlsx.exists():
            rec["emon_summary_xlsx"] = str(out_xlsx)


def _maybe_extract_emon_key_values(*, rows: List[Dict[str, Any]]) -> None:
    # Best-effort: if openpyxl is installed, read the first sheet of summary.xlsx
    # and extract simple key/value pairs (two-column layouts are common).
    try:
        import openpyxl  # type: ignore
    except Exception:
        return

    for rec in rows:
        xlsx = str(rec.get("emon_summary_xlsx") or "").strip()
        if not xlsx:
            continue
        p = Path(xlsx)
        if not p.exists():
            continue
        try:
            wb = openpyxl.load_workbook(p, data_only=True)
            ws = wb[wb.sheetnames[0]]
            kv: Dict[str, Any] = {}
            # Read first 200 rows; look for (key, value) pairs.
            for i, row in enumerate(ws.iter_rows(min_row=1, max_row=200, values_only=True), start=1):
                if not row:
                    continue
                if len(row) < 2:
                    continue
                k = row[0]
                v = row[1]
                if k is None or v is None:
                    continue
                ks = str(k).strip()
                if not ks:
                    continue
                # Avoid huge keys.
                if len(ks) > 80:
                    continue
                if ks in kv:
                    continue
                kv[ks] = v
                if len(kv) >= 80:
                    break
            if kv:
                rec["emon_kv"] = kv
        except Exception:
            continue


def _write_aggregate_csv(*, out_csv: Path, rows: List[Dict[str, Any]]) -> None:
    # Flatten emon_kv: keep it as JSON string to avoid explosion.
    fields = [
        "variant",
        "job_name",
        "backend",
        "model",
        "model_id",
        "batch_size",
        "token_len",
        "tps",
        "latency_sec",
        "avg_batch_time_sec",
        "count",
        "num_batches",
        "exit_code",
        "started_at_utc",
        "ended_at_utc",
        "log_path",
        "metrics_path",
        "emon_enabled",
        "emon_output_path",
        "emon_summary_xlsx",
        "emon_process_rc",
        "emon_process_log",
        "resource_cpu",
        "resource_cpu_count",
        "resource_mem_gb",
        "sglang_max_total_tokens",
        "tps_per_cpu",
        "emon_kv_json",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            emon_kv = r.get("emon_kv")
            r2 = dict(r)
            if emon_kv is not None:
                try:
                    r2["emon_kv_json"] = json.dumps(emon_kv, ensure_ascii=False)
                except Exception:
                    r2["emon_kv_json"] = ""
            else:
                r2["emon_kv_json"] = ""

            # Convenience: allow comparing runs with different CPU caps.
            try:
                cpu_count = int(r2.get("resource_cpu_count") or 0)
            except Exception:
                cpu_count = 0
            try:
                tps = float(r2.get("tps") or 0.0)
            except Exception:
                tps = 0.0
            r2["tps_per_cpu"] = (tps / cpu_count) if cpu_count > 0 else ""

            w.writerow({k: r2.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to scale-test config JSON")
    ap.add_argument("--tee", action="store_true", help="Stream auto-test output to console")
    ap.add_argument("--dry-run", action="store_true", help="Do not execute jobs; print commands only")
    args = ap.parse_args()

    cfg_path_in = Path(args.config)
    cfg_path: Path
    if cfg_path_in.is_absolute():
        cfg_path = cfg_path_in
    else:
        # Prefer resolving relative paths against the user's current working directory.
        # This is important when running from scripts/scale-test/embedding.
        cand_cwd = (Path.cwd() / cfg_path_in).resolve()
        if cand_cwd.exists():
            cfg_path = cand_cwd
        else:
            cand_repo = (REPO_ROOT / cfg_path_in).resolve()
            cfg_path = cand_repo
    if not cfg_path.exists():
        raise SystemExit(
            "Config not found. Tried:\n"
            f"- { (Path.cwd() / cfg_path_in).resolve() }\n"
            f"- { (REPO_ROOT / cfg_path_in).resolve() }\n"
        )

    scale = _parse_scale_config(cfg_path)
    scale_id = _utc_compact()
    out_dir = scale.result_root / scale_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] scale_id={scale_id}")

    cpu_exprs = [str(x or "").strip() for x in (scale.cpu_exprs or [""])]
    if not cpu_exprs:
        cpu_exprs = [""]
    kv_list = [str(x or "").strip() for x in (scale.sglang_max_total_tokens or [""])]
    if not kv_list:
        kv_list = [""]

    single_variant = (len(cpu_exprs) * len(kv_list)) == 1
    variants: List[Dict[str, Any]] = []
    for cpu_expr in cpu_exprs:
        for kv in kv_list:
            variant_name = f"cpu_{_safe_name(cpu_expr or 'none')}__kv_{_safe_name(kv or 'auto')}"
            variant_dir = out_dir if single_variant else (out_dir / variant_name)
            variants.append(
                {
                    "variant": variant_name,
                    "cpu_expr": cpu_expr,
                    "sglang_max_total_tokens": kv,
                    "dir": str(variant_dir),
                }
            )

    if not single_variant:
        _dump_json(out_dir / "variants.json", {"scale_id": scale_id, "variants": variants})

    rows: List[Dict[str, Any]] = []
    for v in variants:
        variant_name = str(v["variant"])
        cpu_expr = str(v.get("cpu_expr") or "")
        kv = str(v.get("sglang_max_total_tokens") or "")
        variant_dir = Path(str(v.get("dir") or ""))
        variant_dir.mkdir(parents=True, exist_ok=True)

        auto_test_cfg = _generate_auto_test_config(
            scale=scale,
            result_dir=variant_dir,
            cpu_expr=cpu_expr,
            sglang_max_total_tokens=kv,
            variant_tag=variant_name if not single_variant else "",
        )
        auto_test_cfg_path = variant_dir / "auto_test_config.generated.json"
        _dump_json(auto_test_cfg_path, auto_test_cfg)

        print(f"[info] variant={variant_name}")
        print(f"[info] auto-test config: {auto_test_cfg_path}")

        cpu_count = _cpu_count(cpu_expr)
        if (cpu_expr or "").strip():
            print(f"[info] CPU expr: {cpu_expr}")
            if cpu_count > 0:
                print(f"[info] CPU count: {cpu_count}")
        if (kv or "").strip():
            print(f"[info] SGLANG_MAX_TOTAL_TOKENS: {kv}")

        rc = _run_auto_test(
            auto_test_config_path=auto_test_cfg_path,
            work_dir=REPO_ROOT,
            tee=bool(args.tee),
            dry_run=bool(args.dry_run),
            cpu_expr=cpu_expr,
            mem_gb=scale.mem_gb,
            stdout_log=variant_dir / "auto_test_stdout.log",
        )
        if rc != 0:
            print(
                f"[error] auto-test runner failed (variant={variant_name}, rc={rc}). "
                f"See {variant_dir / 'auto_test_stdout.log'}"
            )
            if not scale.continue_on_error:
                return rc

        if args.dry_run:
            continue

        summary_found = False
        try:
            run_id = _find_single_run_id(variant_dir)
            summary_csv = variant_dir / f"summary_{run_id}.csv"
            if summary_csv.exists():
                summary_found = True
                for row in _iter_summary_rows(summary_csv):
                    # Pull emon output path from metrics.json (auto-test stores it there).
                    mp = Path(str(row.get("metrics_path") or ""))
                    emon_enabled = ""
                    emon_output_path = ""
                    batch_size = row.get("batch_size") or ""
                    if mp.exists():
                        try:
                            rec = _load_json(mp)
                            emon = rec.get("emon") or {}
                            if isinstance(emon, dict):
                                emon_enabled = str(bool(emon.get("enabled")))
                                emon_output_path = str(emon.get("output_path") or "")
                            env = rec.get("env") or {}
                            if isinstance(env, dict):
                                batch_size = str(env.get("BATCH_SIZE") or batch_size)
                        except Exception:
                            pass

                    merged: Dict[str, Any] = dict(row)
                    merged["variant"] = variant_name
                    merged["emon_enabled"] = emon_enabled
                    merged["emon_output_path"] = emon_output_path
                    merged["batch_size"] = batch_size
                    merged["resource_cpu"] = cpu_expr
                    merged["resource_cpu_count"] = cpu_count if cpu_count > 0 else ""
                    merged["resource_mem_gb"] = scale.mem_gb if scale.mem_gb is not None else ""
                    merged["sglang_max_total_tokens"] = kv
                    rows.append(merged)
        except Exception:
            summary_found = False

        # If the variant failed and we didn't get per-job rows, still record a marker row.
        if rc != 0 and not summary_found:
            rows.append(
                {
                    "variant": variant_name,
                    "job_name": "",
                    "backend": str(scale.job_template.get("backend") or ""),
                    "model": str(scale.job_template.get("model") or ""),
                    "model_id": str(scale.job_template.get("model_id") or ""),
                    "batch_size": "",
                    "token_len": "",
                    "tps": "",
                    "latency_sec": "",
                    "avg_batch_time_sec": "",
                    "count": "",
                    "num_batches": "",
                    "exit_code": str(rc),
                    "started_at_utc": "",
                    "ended_at_utc": "",
                    "log_path": "",
                    "metrics_path": "",
                    "emon_enabled": "",
                    "emon_output_path": "",
                    "emon_summary_xlsx": "",
                    "emon_process_rc": "",
                    "emon_process_log": "",
                    "resource_cpu": cpu_expr,
                    "resource_cpu_count": cpu_count if cpu_count > 0 else "",
                    "resource_mem_gb": scale.mem_gb if scale.mem_gb is not None else "",
                    "sglang_max_total_tokens": kv,
                }
            )

    if args.dry_run:
        print("[ok] dry-run complete (no results generated)")
        return 0

    if scale.emon_process_after_run and rows:
        _process_emon_dirs(
            rows=rows,
            tee=bool(args.tee),
            process_cmd=scale.emon_process_cmd,
            expected_output=scale.emon_expected_output,
        )
        _maybe_extract_emon_key_values(rows=rows)

    agg_csv = out_dir / "aggregate.csv"
    _write_aggregate_csv(out_csv=agg_csv, rows=rows)
    print(f"[ok] Wrote aggregate: {agg_csv}")

    if single_variant:
        run_id = _find_single_run_id(out_dir)
        summary_csv = out_dir / f"summary_{run_id}.csv"
        if summary_csv.exists():
            print(f"[ok] Auto-test summary: {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
