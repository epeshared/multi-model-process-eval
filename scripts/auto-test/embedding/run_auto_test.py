#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.parse
import urllib.request
import urllib.parse


REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_cpu_list_expr(expr: str) -> List[int]:
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
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(part))
            except Exception:
                continue
    return sorted(set(out))


def _read_self_status_fields() -> Dict[str, str]:
    try:
        p = Path("/proc/self/status")
        if not p.exists():
            return {}
        fields: Dict[str, str] = {}
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Cpus_allowed_list:") or line.startswith("Mems_allowed_list:") or line.startswith("Cpus_allowed:"):
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
        return fields
    except Exception:
        return {}


def _apply_self_affinity_from_env(*, tee: bool) -> None:
    expr = str(os.environ.get("AUTO_TEST_CPU_EXPR") or "").strip()
    if not expr:
        return

    before = _read_self_status_fields()
    cpus = _parse_cpu_list_expr(expr)
    if not cpus:
        print(f"[affinity] AUTO_TEST_CPU_EXPR set but unparsable: {expr}")
        return

    try:
        os.sched_setaffinity(0, set(cpus))
    except Exception as e:
        print(f"[affinity] failed to apply AUTO_TEST_CPU_EXPR={expr}: {e}")
        return

    after = _read_self_status_fields()
    # Always print one evidence line so remote logs show binding.
    print(
        "[affinity] applied "
        + f"AUTO_TEST_CPU_EXPR={expr} "
        + f"before={before.get('Cpus_allowed_list', '?')} "
        + f"after={after.get('Cpus_allowed_list', '?')}"
    )

    if tee and shutil.which("taskset"):
        try:
            p = subprocess.run(
                ["taskset", "-pc", str(os.getpid())],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            out = (p.stdout or "").strip()
            if out:
                print(f"[affinity] {out}")
        except Exception:
            pass


@dataclass(frozen=True)
class Job:
    name: str
    script: str
    args: List[str]
    env: Dict[str, str]
    timeout_sec: Optional[float]
    warmup_runs: int
    repeats: int
    repeat_threshold_sec: Optional[float]
    repeat_max_repeats: int
    restart_servers: bool
    stop_server_after_job: bool
    emon_enable: bool


def _parse_int_ge(v: Any, *, default: int, min_value: int, field_name: str) -> int:
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        iv = int(float(v))
    except Exception:
        raise SystemExit(f"{field_name} must be an integer >= {min_value}")
    if iv < min_value:
        raise SystemExit(f"{field_name} must be an integer >= {min_value}")
    return int(iv)


def _to_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    mu = sum(values) / float(len(values))
    var = sum((x - mu) ** 2 for x in values) / float(len(values))
    return mu, var ** 0.5


def _aggregate_repeat_metrics(
    *,
    script: str,
    per_repeat: List[Dict[str, Any]],
    include_failed: bool = False,
) -> Dict[str, Any]:
    """Aggregate parsed metrics across repeats.

    - Uses only repeats with exit_code==0 by default.
    - Returns a metrics dict compatible with existing CSV fields (tps/latency_sec/etc).
    """

    ok_recs: List[Dict[str, Any]] = []
    for r in per_repeat:
        if include_failed:
            ok_recs.append(r)
        else:
            if int(r.get("exit_code") or 0) == 0:
                ok_recs.append(r)

    out: Dict[str, Any] = {
        "repeats": int(len(per_repeat)),
        "ok_repeats": int(len(ok_recs)),
        "per_repeat": per_repeat,
    }

    if not ok_recs:
        out["parse_error"] = "no_successful_repeats"
        return out

    # Average common numeric metrics across successful repeats.
    numeric_keys = [
        "tps",
        "latency_sec",
        "avg_batch_time_sec",
        "count",
        "num_batches",
        "qps",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "max_concurrency",
    ]
    for k in numeric_keys:
        vals: List[float] = []
        for r in ok_recs:
            v = _to_float_or_none((r.get("metrics") or {}).get(k))
            if v is not None:
                vals.append(float(v))
        mean_v, std_v = _mean_std(vals)
        if mean_v is not None:
            out[k] = mean_v
            out[f"{k}_std"] = std_v

    # Keep one representative summary for debugging (use the last successful one).
    last_metrics = ok_recs[-1].get("metrics") or {}
    if isinstance(last_metrics, dict):
        if "summary" in last_metrics:
            out["summary"] = last_metrics.get("summary")

    return out


def _parse_bool(v: Any, *, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass
class EmonSession:
    proc: subprocess.Popen[str]
    out_f: Any
    output_path: Path


def _start_emon_session(*, output_path: Path, tee: bool, prefix: str = "") -> EmonSession:
    emon_bin = shutil.which("emon")
    if not emon_bin:
        raise SystemExit("emon_enable is true but 'emon' was not found in PATH")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = output_path.open("w", encoding="utf-8")

    if tee:
        print(f"{prefix}[emon] start: {emon_bin} -collect-edp > {output_path}")

    proc = subprocess.Popen(
        [emon_bin, "-collect-edp"],
        cwd=str(output_path.parent),
        stdout=out_f,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    return EmonSession(proc=proc, out_f=out_f, output_path=output_path)


def _stop_emon_session(session: EmonSession, *, tee: bool, prefix: str = "") -> None:
    emon_bin = shutil.which("emon")
    if not emon_bin:
        try:
            session.out_f.close()
        except Exception:
            pass
        return

    try:
        if tee:
            print(f"{prefix}[emon] stop: {emon_bin} -stop")
        try:
            session.out_f.flush()
        except Exception:
            pass

        subprocess.run(
            [emon_bin, "-stop"],
            cwd=str(session.output_path.parent),
            stdout=session.out_f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        try:
            session.proc.wait(timeout=10.0)
        except Exception:
            pass
    finally:
        try:
            session.out_f.close()
        except Exception:
            pass


@dataclass(frozen=True)
class ServerSpec:
    backend: str
    start_script: str
    cwd: str
    args: List[str]
    env: Dict[str, str]
    env_from_job: Dict[str, str]
    enabled: bool
    ready_endpoints: List[str]
    ready_timeout_sec: float
    ready_interval_sec: float
    numactl_cores: str
    numactl_cpunodebind: str
    numactl_membind: str


@dataclass
class RunningServer:
    spec: ServerSpec
    base_url: str
    proc: Optional[subprocess.Popen[str]]
    started_by_runner: bool
    log_path: Path
    tee_thread: Optional[threading.Thread] = None
    numactl_cores: str = ""
    numactl_cpunodebind: str = ""
    numactl_membind: str = ""


def _effective_numactl_for_server(*, spec: ServerSpec, job_env: Dict[str, str]) -> Tuple[str, str, str]:
    # NOTE: Keep server NUMA binding independent from job NUMA binding.
    #
    # - Job-level NUMACTL_* is reserved for the benchmark/job process itself.
    # - Server binding defaults come from servers.<backend>.numactl in the config.
    # - If you need to override server binding per-job, use SERVER_NUMACTL_*.
    numactl_cores = (job_env.get("SERVER_NUMACTL_CORES") or "").strip() or (spec.numactl_cores or "").strip()
    numactl_cpunodebind = (job_env.get("SERVER_NUMACTL_CPUNODEBIND") or "").strip() or (spec.numactl_cpunodebind or "").strip()
    numactl_membind = (job_env.get("SERVER_NUMACTL_MEMBIND") or "").strip() or (spec.numactl_membind or "").strip()
    return numactl_cores, numactl_cpunodebind, numactl_membind


def _effective_numactl_for_job(*, job_env: Dict[str, str]) -> Tuple[str, str, str]:
    """Return NUMA binding for the benchmark/job process.

    Uses job-level NUMACTL_* env vars.
    """

    numactl_cores = (job_env.get("NUMACTL_CORES") or "").strip()
    numactl_cpunodebind = (job_env.get("NUMACTL_CPUNODEBIND") or "").strip()
    numactl_membind = (job_env.get("NUMACTL_MEMBIND") or "").strip()
    return numactl_cores, numactl_cpunodebind, numactl_membind


def _maybe_wrap_cmd_with_numactl(
    *,
    cmd: List[str],
    numactl_cores: str,
    numactl_cpunodebind: str,
    numactl_membind: str,
) -> List[str]:
    """Wrap a command with numactl if any binding fields are provided."""

    if not (numactl_cores or numactl_cpunodebind or numactl_membind):
        return cmd
    numactl_bin = shutil.which("numactl")
    if not numactl_bin:
        return cmd

    prefix: List[str] = [numactl_bin]
    if numactl_cores:
        prefix += ["-C", numactl_cores]
    if numactl_cpunodebind:
        prefix += ["--cpunodebind", numactl_cpunodebind]
    if numactl_membind:
        prefix += ["--membind", numactl_membind]
    else:
        # IMPORTANT:
        # If the parent process is started under a restrictive memory policy
        # (e.g. `numactl -m 2`), that policy is inherited by child processes.
        # When we apply CPU affinity via numactl without a memory policy, the
        # inherited membind can cause OOM-killer events even though the machine
        # has plenty of RAM.
        #
        # Default to an explicit, non-restrictive policy that matches the CPU
        # binding. Users can opt out by setting:
        #   AUTO_TEST_NUMACTL_DEFAULT_MEMPOLICY=inherit
        default_policy = (os.environ.get("AUTO_TEST_NUMACTL_DEFAULT_MEMPOLICY") or "localalloc").strip().lower()
        if default_policy in {"localalloc", "local"}:
            prefix += ["--localalloc"]
        elif default_policy in {"interleave", "interleave=all", "all"}:
            prefix += ["--interleave=all"]
    prefix += ["--"]
    return prefix + cmd


def _parse_cpu_list_expr(expr: str) -> List[int]:
    """Parse a linux CPU list expression like '0-15,32,40-47' into a sorted list."""

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
            # Guard against absurd ranges.
            span = min(hi - lo, 100000)
            for x in range(lo, lo + span + 1):
                out.append(x)
        else:
            try:
                out.append(int(part))
            except Exception:
                continue
    # Dedupe
    return sorted(set(out))


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


def _infer_single_numa_node_from_cores(cores_expr: str) -> Optional[int]:
    """Infer a single NUMA node id if all requested cores map to the same node."""

    cpus = _parse_cpu_list_expr(cores_expr)
    if not cpus:
        return None

    nodes: List[int] = []
    for cpu in cpus[:256]:
        n = _cpu_to_numa_node(cpu)
        if n is None:
            # If sysfs can't map a cpu, don't guess.
            return None
        nodes.append(n)

    uniq = sorted(set(nodes))
    if len(uniq) == 1:
        return uniq[0]
    return None


def _read_numa_node_cpulist(numa_node: int) -> Optional[str]:
    """Read /sys NUMA node cpulist (e.g. '0-95,192-287')."""

    try:
        p = Path(f"/sys/devices/system/node/node{int(numa_node)}/cpulist")
        s = p.read_text(encoding="utf-8").strip()
        return s or None
    except Exception:
        return None


def _infer_sglang_cpu_omp_threads_bind(
    *,
    numactl_cores: str,
    numactl_cpunodebind: str,
    numactl_membind: str,
) -> Optional[str]:
    """Infer a safe SGLANG_CPU_OMP_THREADS_BIND value.

    For sglang CPU+AMX, `torch.ops.sgl_kernel.init_cpu_threads_env()` will pin
    OpenMP threads to `local_omp_cpuid`. If we externally bind the server with
    numactl (e.g. to node1), leaving SGLANG_CPU_OMP_THREADS_BIND unset makes
    sglang default to NUMA node0 for tp_rank=0, which can crash when the process
    is not allowed to run on those CPUs.

    We align thread binding with the runner's server numactl binding.
    """

    cores = (numactl_cores or "").strip()
    if cores:
        return cores

    node: Optional[int] = None
    if (numactl_cpunodebind or "").strip().isdigit():
        node = int(numactl_cpunodebind)
    elif (numactl_membind or "").strip().isdigit():
        node = int(numactl_membind)

    if node is None or node < 0:
        return None

    return _read_numa_node_cpulist(node)


def _teardown_server(rs: RunningServer) -> None:
    if not rs.started_by_runner:
        return
    proc = rs.proc
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=10.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    t = rs.tee_thread
    if t is not None:
        try:
            t.join(timeout=2.0)
        except Exception:
            pass


def _utc_now_compact() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _merge_env(base: Dict[str, str], override: Dict[str, str]) -> Dict[str, str]:
    out = dict(base)
    out.update({k: str(v) for k, v in override.items()})
    return out


def _norm_backend(backend: str) -> str:
    b = (backend or "").strip().lower()
    # Normalize a few common aliases.
    if b in {"vllm", "vllm_openai", "vllm-http-openai", "vllm_openai_http"}:
        return "vllm-http"
    if b in {"sglang-http", "sglang_http"}:
        return "sglang"
    return b


def _norm_base_url(base_url: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        return ""
    if b.startswith("http://") or b.startswith("https://"):
        return b
    return f"http://{b}"


def _url_is_local(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).hostname
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}


def _parse_host_port(base_url: str) -> Tuple[str, int]:
    u = urllib.parse.urlparse(_norm_base_url(base_url))
    host = u.hostname or ""
    port = int(u.port or 0)
    return host, port


def _list_listening_pids(port: int) -> List[int]:
    if port <= 0:
        return []
    ss_bin = shutil.which("ss")
    if not ss_bin:
        return []
    try:
        proc = subprocess.run(
            [ss_bin, "-ltnp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3.0,
        )
    except Exception:
        return []

    out = proc.stdout or ""
    # Example snippet: users:(("python",pid=12345,fd=7))
    pids: List[int] = []
    for line in out.splitlines():
        if f":{port}" not in line:
            continue
        for m in re.finditer(r"pid=(\d+)", line):
            try:
                pids.append(int(m.group(1)))
            except Exception:
                pass
    # Dedupe while preserving order
    seen = set()
    uniq: List[int] = []
    for pid in pids:
        if pid not in seen:
            seen.add(pid)
            uniq.append(pid)
    return uniq


def _print_pid_affinity_debug(*, pid: int, prefix: str) -> None:
    """Print best-effort CPU pinning evidence for a given pid."""

    try:
        proc = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "pid,ppid,etime,%cpu,psr,nlwp,cmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2.0,
            check=False,
        )
        out = (proc.stdout or "").strip()
        if out:
            for line in out.splitlines():
                print(prefix + line)
    except Exception:
        pass

    taskset_bin = shutil.which("taskset")
    if taskset_bin:
        try:
            proc = subprocess.run(
                [taskset_bin, "-pc", str(int(pid))],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=2.0,
                check=False,
            )
            out = (proc.stdout or "").strip()
            if out:
                print(prefix + out)
        except Exception:
            pass

    try:
        status_path = Path(f"/proc/{int(pid)}/status")
        if status_path.exists():
            text = status_path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("Cpus_allowed_list:") or line.startswith("Mems_allowed_list:"):
                    print(prefix + line)
    except Exception:
        pass

    try:
        cgroup_path = Path(f"/proc/{int(pid)}/cgroup")
        if cgroup_path.exists():
            cg = cgroup_path.read_text(encoding="utf-8", errors="replace").strip()
            if cg:
                # Usually one line on cgroup v2: "0::/path"
                print(prefix + "cgroup=" + cg.splitlines()[0])
    except Exception:
        pass


def _print_server_pid_debug(*, base_url: str, backend: str, proc: Optional[subprocess.Popen[str]]) -> None:
    """Print server PID + pinning evidence after server is ready."""

    base_url = _norm_base_url(base_url)
    if not base_url:
        return

    pids: List[int] = []
    if proc is not None and getattr(proc, "pid", None):
        try:
            pids = [int(proc.pid)]
        except Exception:
            pids = []
    else:
        # External server: try to resolve PID by local listening port.
        try:
            host, port = _parse_host_port(base_url)
            if host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"} and port > 0:
                pids = _list_listening_pids(port)
        except Exception:
            pids = []

    if not pids:
        print(f"[server:{backend}] pid_debug: unable to resolve server pid for {base_url}")
        return

    # Keep output short if multiple PIDs are present.
    show = pids[:3]
    more = len(pids) - len(show)
    print(f"[server:{backend}] pid_debug: base_url={base_url} pids={show}{' (+%d more)' % more if more > 0 else ''}")
    for pid in show:
        _print_pid_affinity_debug(pid=pid, prefix=f"[server:{backend}] ")


def _try_shutdown_existing_listener(*, base_url: str, tee: bool) -> bool:
    """Best-effort shutdown of any local process listening on base_url's port.

    Returns True if it attempted to signal at least one PID.
    """

    host, port = _parse_host_port(base_url)
    if port <= 0:
        return False
    if host not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        # Don't try to kill non-local hosts.
        return False

    pids = [pid for pid in _list_listening_pids(port) if pid != os.getpid()]
    if not pids:
        return False

    if tee:
        print(f"[server] restart requested; shutting down listeners on :{port} (pids={pids})")

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not _list_listening_pids(port):
            return True
        time.sleep(0.2)

    # Escalate.
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    return True


def _urlopen_no_proxy_for_local(req: urllib.request.Request, *, timeout_sec: float) -> Any:
    """Open URL requests, bypassing env proxies for localhost.

    Some environments set http(s)_proxy and urllib may route localhost probes via proxy,
    leading to confusing 403/connection behavior and false readiness detection.
    """

    if _url_is_local(req.full_url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=timeout_sec)
    return urllib.request.urlopen(req, timeout=timeout_sec)


def _http_get_ok(url: str, *, timeout_sec: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with _urlopen_no_proxy_for_local(req, timeout_sec=timeout_sec) as resp:
            code = getattr(resp, "status", None) or 200
            return 200 <= int(code) < 300
    except urllib.error.HTTPError as e:
        # Consider 401/403 as "up" for model endpoints.
        if int(getattr(e, "code", 0) or 0) in {401, 403}:
            return True
        return False
    except urllib.error.URLError:
        return False
    except Exception:
        return False


def _http_post_json_ok(url: str, payload: Dict[str, Any], *, timeout_sec: float = 3.0) -> bool:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with _urlopen_no_proxy_for_local(req, timeout_sec=timeout_sec) as resp:
            code = getattr(resp, "status", None) or 200
            return 200 <= int(code) < 300
    except urllib.error.HTTPError as e:
        code = int(getattr(e, "code", 0) or 0)
        # If the endpoint exists but payload is rejected, server is still up.
        if 400 <= code < 500 and code != 404:
            return True
        return False
    except urllib.error.URLError:
        return False
    except Exception:
        return False


def _wait_server_ready(
    *,
    base_url: str,
    endpoints: List[str],
    timeout_sec: float,
    interval_sec: float,
    probe_model_id: str = "",
    proc: Optional[subprocess.Popen[str]] = None,
) -> bool:
    base_url = _norm_base_url(base_url)
    if not base_url:
        return False

    eps = endpoints or ["POST /v1/embeddings", "/v1/models", "/health"]
    deadline = time.time() + max(1.0, float(timeout_sec))
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            # Server process has exited; don't wait out the full timeout.
            return False
        for ep in eps:
            ep_s = str(ep or "").strip()
            if not ep_s:
                continue

            m = re.match(r"^(GET|POST)\s+(.+)$", ep_s, flags=re.IGNORECASE)
            method = "GET"
            path = ep_s
            if m:
                method = (m.group(1) or "GET").upper()
                path = (m.group(2) or "").strip()

            if not path.startswith("/"):
                path = "/" + path

            url = f"{base_url}{path}"
            if method == "POST":
                # Default readiness probe for OpenAI-compatible embedding servers.
                payload: Dict[str, Any] = {
                    "model": probe_model_id or "default",
                    "input": ["ping"],
                }
                if path.startswith("/v1/"):
                    payload.setdefault("encoding_format", "float")
                if _http_post_json_ok(url, payload):
                    return True
            else:
                if _http_get_ok(url):
                    return True
        time.sleep(max(0.1, float(interval_sec)))
    return False


def _tail_file(path: Path, *, max_lines: int = 80) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    if len(lines) <= int(max_lines):
        return text
    return "\n".join(lines[-int(max_lines) :]) + "\n"


def _parse_servers(cfg: Dict[str, Any]) -> Dict[str, ServerSpec]:
    servers_cfg = cfg.get("servers") or {}
    out: Dict[str, ServerSpec] = {}
    if not isinstance(servers_cfg, dict):
        return out

    for backend_key, raw in servers_cfg.items():
        if not isinstance(raw, dict):
            continue
        backend = _norm_backend(str(backend_key))
        enabled = bool(raw.get("enabled", True))
        start_script = str(raw.get("start_script") or "").strip()
        if not start_script:
            continue
        cwd = str(raw.get("cwd") or ".").strip()
        args = raw.get("args") or []
        if not isinstance(args, list):
            raise SystemExit(f"servers.{backend_key}.args must be a list")
        env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()} if isinstance(raw.get("env"), dict) else {}
        env_from_job = (
            {str(k): str(v) for k, v in (raw.get("env_from_job") or {}).items()}
            if isinstance(raw.get("env_from_job"), dict)
            else {}
        )
        ready = raw.get("ready") or {}
        ready_endpoints = ready.get("endpoints") if isinstance(ready, dict) else None
        if ready_endpoints is None:
            ready_endpoints = ["/v1/models", "/health"]
        if not isinstance(ready_endpoints, list):
            raise SystemExit(f"servers.{backend_key}.ready.endpoints must be a list")
        ready_timeout_sec = float(ready.get("timeout_sec", 900)) if isinstance(ready, dict) else 900.0
        ready_interval_sec = float(ready.get("interval_sec", 2)) if isinstance(ready, dict) else 2.0

        numactl_cfg = raw.get("numactl") or {}
        numactl_cores = ""
        numactl_cpunodebind = ""
        numactl_membind = ""
        if isinstance(numactl_cfg, dict):
            numactl_cores = str(numactl_cfg.get("cores") or "").strip()
            numactl_cpunodebind = str(numactl_cfg.get("cpunodebind") or "").strip()
            numactl_membind = str(numactl_cfg.get("membind") or "").strip()
        else:
            # Allow shorthand: "numactl": "0-15"
            numactl_cores = str(numactl_cfg).strip()

        out[backend] = ServerSpec(
            backend=backend,
            start_script=start_script,
            cwd=cwd,
            args=[str(a) for a in args],
            env=env,
            env_from_job=env_from_job,
            enabled=enabled,
            ready_endpoints=[str(x) for x in ready_endpoints],
            ready_timeout_sec=ready_timeout_sec,
            ready_interval_sec=ready_interval_sec,
            numactl_cores=numactl_cores,
            numactl_cpunodebind=numactl_cpunodebind,
            numactl_membind=numactl_membind,
        )

    return out


def _server_key(spec: ServerSpec, base_url: str) -> str:
    return f"{spec.backend}|{_norm_base_url(base_url)}"


def _ensure_server(
    *,
    spec: ServerSpec,
    job: Job,
    base_url: str,
    result_dir: Path,
    run_id: str,
    idx: int,
    dry_run: bool,
    tee: bool,
    restart_servers: bool,
    running: Dict[str, RunningServer],
) -> Optional[RunningServer]:
    if not spec.enabled:
        return None

    base_url = _norm_base_url(base_url)
    if not base_url:
        return None

    key = _server_key(spec, base_url)
    if key in running:
        return running[key]

    eff_cores, eff_cpunodebind, eff_membind = _effective_numactl_for_server(spec=spec, job_env=job.env)

    if tee:
        print(f"[server:{spec.backend}] checking ready: {base_url}")

    if restart_servers and not dry_run:
        # Best-effort: kill any leftover local server on this port, even if it isn't ready.
        _try_shutdown_existing_listener(base_url=base_url, tee=tee)

    # If server is already up, don't start a new one.
    probe_model_id = (
        job.env.get("MODEL_ID")
        or job.env.get("SERVED_MODEL_NAME")
        or job.env.get("MODEL")
        or ""
    )
    if _wait_server_ready(
        base_url=base_url,
        endpoints=spec.ready_endpoints,
        timeout_sec=0.5,
        interval_sec=0.1,
        probe_model_id=probe_model_id,
    ):
        if restart_servers and not dry_run:
            raise SystemExit(
                f"restart_servers requested but server is still responding at {base_url}. "
                "Please stop it manually or disable restart_servers for this job."
            )

        if tee:
            print(f"[server:{spec.backend}] already running at {base_url} (external); will not start")

        # Hook: print CPU pinning evidence for the external server too.
        _print_server_pid_debug(base_url=base_url, backend=spec.backend, proc=None)

        rs = RunningServer(
            spec=spec,
            base_url=base_url,
            proc=None,
            started_by_runner=False,
            log_path=result_dir / f"{run_id}_{idx:03d}_server_{_sanitize_filename(spec.backend)}.external.log",
            numactl_cores=eff_cores,
            numactl_cpunodebind=eff_cpunodebind,
            numactl_membind=eff_membind,
        )
        running[key] = rs
        return rs

    script_path = (REPO_ROOT / spec.start_script).resolve()
    if not script_path.exists():
        raise SystemExit(f"Server start_script not found: {script_path}")

    cwd = (REPO_ROOT / spec.cwd).resolve() if not Path(spec.cwd).is_absolute() else Path(spec.cwd)
    env = os.environ.copy()
    env.update(spec.env)
    for env_key, job_key in spec.env_from_job.items():
        if job_key in job.env:
            env[env_key] = str(job.env[job_key])

    # If the user doesn't explicitly provide SGLANG_PYTHON, default it to the
    # current runner's interpreter. This makes JSON configs portable across
    # hosts (especially for remote dispatch) as long as the runner is executed
    # under the desired conda env.
    if (
        spec.backend == "sglang"
        and not str(env.get("SGLANG_PYTHON") or "").strip()
        and not str(env.get("SGLANG_CONDA_ENV") or "").strip()
    ):
        env["SGLANG_PYTHON"] = sys.executable

    # For sglang CPU+AMX: align sglang's OpenMP thread binding with our server
    # NUMA binding to avoid pinning to disallowed CPUs (which can segfault in
    # sgl_kernel.init_cpu_threads_env).
    if spec.backend == "sglang" and "SGLANG_CPU_OMP_THREADS_BIND" not in env:
        inferred_bind = _infer_sglang_cpu_omp_threads_bind(
            numactl_cores=eff_cores,
            numactl_cpunodebind=eff_cpunodebind,
            numactl_membind=eff_membind,
        )
        if inferred_bind:
            env["SGLANG_CPU_OMP_THREADS_BIND"] = inferred_bind

    # sglang can be passed a NUMA node hint via SGLANG_NUMA_NODE. However, doing
    # this implicitly can change how sglang wraps its subprocesses (it may add
    # internal numactl wrappers). Keep this opt-in.
    if spec.backend == "sglang" and "SGLANG_NUMA_NODE" not in env:
        if _parse_bool(os.environ.get("AUTO_SGLANG_NUMA_NODE"), default=False):
            inferred_node: Optional[int] = None

            # Prefer explicit node binding if provided.
            if eff_cpunodebind.isdigit():
                inferred_node = int(eff_cpunodebind)
            elif eff_membind.isdigit():
                inferred_node = int(eff_membind)
            else:
                inferred_node = _infer_single_numa_node_from_cores(eff_cores)

            if inferred_node is not None and inferred_node >= 0:
                env["SGLANG_NUMA_NODE"] = str(inferred_node)

    log_path = result_dir / f"{run_id}_{idx:03d}_server_{_sanitize_filename(spec.backend)}.log"

    cmd = ["bash", str(script_path)] + list(spec.args)

    # Optional NUMA binding for server process.
    numactl_cores, numactl_cpunodebind, numactl_membind = eff_cores, eff_cpunodebind, eff_membind

    # IMPORTANT: do not implicitly infer --cpunodebind/--membind.
    # If the user leaves them empty, we start without them.

    if numactl_cores or numactl_cpunodebind or numactl_membind:
        numactl_bin = shutil.which("numactl")
        if not numactl_bin:
            warn_bits = []
            if numactl_cores:
                warn_bits.append(f"SERVER_NUMACTL_CORES={numactl_cores}")
            if numactl_cpunodebind:
                warn_bits.append(f"SERVER_NUMACTL_CPUNODEBIND={numactl_cpunodebind}")
            if numactl_membind:
                warn_bits.append(f"SERVER_NUMACTL_MEMBIND={numactl_membind}")
            print(f"[warn] numactl requested ({', '.join(warn_bits)}) but numactl not found; starting without binding")
        else:
            prefix: List[str] = [numactl_bin]
            if numactl_cores:
                prefix += ["-C", numactl_cores]
            if numactl_cpunodebind:
                prefix += ["--cpunodebind", numactl_cpunodebind]
            if numactl_membind:
                prefix += ["--membind", numactl_membind]
            else:
                default_policy = (os.environ.get("AUTO_TEST_NUMACTL_DEFAULT_MEMPOLICY") or "localalloc").strip().lower()
                if default_policy in {"localalloc", "local"}:
                    prefix += ["--localalloc"]
                elif default_policy in {"interleave", "interleave=all", "all"}:
                    prefix += ["--interleave=all"]
            prefix += ["--"]
            cmd = prefix + cmd

    if dry_run:
        print(f"[dry-run] start-server backend={spec.backend} base_url={base_url}: {' '.join(cmd)}")
        rs = RunningServer(
            spec=spec,
            base_url=base_url,
            proc=None,
            started_by_runner=True,
            log_path=log_path,
            numactl_cores=eff_cores,
            numactl_cpunodebind=eff_cpunodebind,
            numactl_membind=eff_membind,
        )
        running[key] = rs
        return rs

    if tee:
        print(f"[server:{spec.backend}] starting (log: {log_path})")
        print(f"[server:{spec.backend}] cmd: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    tee_thread: Optional[threading.Thread] = None
    if proc.stdout is not None:
        tee_thread = threading.Thread(
            target=_pump_stream_to_log,
            kwargs={
                "stream": proc.stdout,
                "log_path": log_path,
                "tee": tee,
                "prefix": f"[server:{spec.backend}] ",
            },
            daemon=True,
        )
        tee_thread.start()

    ok = _wait_server_ready(
        base_url=base_url,
        endpoints=spec.ready_endpoints,
        timeout_sec=spec.ready_timeout_sec,
        interval_sec=spec.ready_interval_sec,
        probe_model_id=probe_model_id,
        proc=proc,
    )
    if not ok:
        rc = proc.poll()
        if rc is not None:
            # Join tee thread briefly so logs are flushed before tailing.
            if tee_thread is not None:
                try:
                    tee_thread.join(timeout=2.0)
                except Exception:
                    pass

            tail = _tail_file(log_path, max_lines=120)
            msg = (
                f"Server exited before becoming ready: backend={spec.backend} base_url={base_url} exit_code={rc} "
                f"(see {log_path})"
            )
            if tail.strip():
                msg += "\n--- last log lines ---\n" + tail
            raise SystemExit(msg)

        try:
            proc.terminate()
        except Exception:
            pass
        raise SystemExit(f"Server failed to become ready: backend={spec.backend} base_url={base_url} (see {log_path})")

    rs = RunningServer(
        spec=spec,
        base_url=base_url,
        proc=proc,
        started_by_runner=True,
        log_path=log_path,
        tee_thread=tee_thread,
        numactl_cores=eff_cores,
        numactl_cpunodebind=eff_cpunodebind,
        numactl_membind=eff_membind,
    )

    # Hook: after server is ready, print pid + affinity/cpuset evidence.
    _print_server_pid_debug(base_url=base_url, backend=spec.backend, proc=proc)

    running[key] = rs
    return rs


def _teardown_servers(running: Dict[str, RunningServer]) -> None:
    # Stop in reverse start order for safety.
    items = [v for v in running.values() if v.started_by_runner and v.proc is not None]
    for rs in reversed(items):
        proc = rs.proc
        if proc is None:
            continue
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    # Wait then kill.
    deadline = time.time() + 10.0
    for rs in reversed(items):
        proc = rs.proc
        if proc is None:
            continue
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    for rs in reversed(items):
        t = rs.tee_thread
        if t is None:
            continue
        try:
            t.join(timeout=2.0)
        except Exception:
            pass


def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "job"


def _pump_stream_to_log(
    *,
    stream: Any,
    log_path: Path,
    tee: bool,
    prefix: str,
) -> None:
    """Continuously read text from a stream, write to log, optionally echo to stdout."""

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_f:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                log_f.write(line)
                log_f.flush()
                if tee:
                    sys.stdout.write(prefix + line)
                    sys.stdout.flush()
    except Exception:
        # Don't let logging failures crash the whole run.
        return
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _run_job_streaming(
    *,
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    tee: bool,
    prefix: str,
    timeout_sec: Optional[float] = None,
) -> Tuple[int, str, float]:
    """Run a job, writing logs incrementally and returning full combined stdout for parsing."""

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    chunks: List[str] = []
    assert proc.stdout is not None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                chunks.append(line)
                log_f.write(line)
                log_f.flush()
                if tee:
                    sys.stdout.write(prefix + line)
                    sys.stdout.flush()
            rc = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            rc = 124
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    return rc, "".join(chunks), time.time() - start


def _extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the last JSON object in a blob of text.

    This is tailored to scripts/embedding/run_embedding.py which prints a JSON object
    at the end (pretty-printed).
    """

    # Fast path: try to find the last '{' that begins a JSON object and parse progressively.
    # We scan all brace-balanced blocks and keep the last that parses.
    last_obj: Optional[Dict[str, Any]] = None

    starts = [m.start() for m in re.finditer(r"\{", text)]
    if not starts:
        return None

    for start in starts:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            last_obj = obj
                    except Exception:
                        pass
                    break

    return last_obj


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract all JSON objects from a blob of text (best-effort)."""

    objs: List[Dict[str, Any]] = []
    starts = [m.start() for m in re.finditer(r"\{", text)]
    if not starts:
        return objs

    for start in starts:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            objs.append(obj)
                    except Exception:
                        pass
                    break

    return objs


def _pick_embedding_summary(objs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not objs:
        return None

    # Prefer objects that look like the final embedding summary.
    # scripts/embedding/run_embedding.py prints a dict containing at least:
    #   tps, time_sec, count, num_batches, avg_batch_time_sec
    def looks_like_summary(o: Dict[str, Any]) -> bool:
        if "tps" in o and "time_sec" in o:
            return True
        if "count" in o and "time_sec" in o and "num_batches" in o:
            return True
        return False

    for o in reversed(objs):
        if looks_like_summary(o):
            return o

    # Fallback: return the largest dict to avoid picking tiny nested dicts.
    return max(objs, key=lambda d: len(d.keys()))


def _parse_embedding_metrics(stdout_stderr: str) -> Dict[str, Any]:
    objs = _extract_json_objects(stdout_stderr)
    obj = _pick_embedding_summary(objs) if objs else _extract_last_json_object(stdout_stderr)
    if not obj:
        return {"parse_error": "no_json_summary_found"}

    # Normalize fields across datasets
    metrics: Dict[str, Any] = {
        "summary": obj,
    }

    # Common embedding summary
    if "tps" in obj and "time_sec" in obj:
        metrics["tps"] = obj.get("tps")
        metrics["latency_sec"] = obj.get("time_sec")
        metrics["avg_batch_time_sec"] = obj.get("avg_batch_time_sec")
        metrics["count"] = obj.get("count")
        metrics["num_batches"] = obj.get("num_batches")
        return metrics

    # If we found a dict but it doesn't match expected schema, keep it for debugging.
    metrics["parse_error"] = "unrecognized_summary_schema"
    return metrics

    # Flickr8k produces modality-specific metrics
    if "text_tps" in obj or "image_tps" in obj:
        if "text_tps" in obj:
            metrics["tps"] = obj.get("text_tps")
            metrics["latency_sec"] = obj.get("text_time_sec")
            metrics["avg_batch_time_sec"] = obj.get("text_avg_batch_time_sec")
            metrics["count"] = obj.get("text_count")
            metrics["num_batches"] = obj.get("text_num_batches")
        elif "image_tps" in obj:
            metrics["tps"] = obj.get("image_tps")
            metrics["latency_sec"] = obj.get("image_time_sec")
            metrics["avg_batch_time_sec"] = obj.get("image_avg_batch_time_sec")
            metrics["count"] = obj.get("image_count")
            metrics["num_batches"] = obj.get("image_num_batches")
        return metrics

    return {"summary": obj, "parse_error": "unrecognized_summary_schema"}


def _extract_prefixed_json_line(text: str, prefix: str) -> Optional[Dict[str, Any]]:
    pat = re.compile(rf"^{re.escape(prefix)}=(.+)$", re.MULTILINE)
    last_obj: Optional[Dict[str, Any]] = None
    for m in pat.finditer(text):
        payload = (m.group(1) or "").strip()
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict):
            last_obj = obj
    return last_obj


def _parse_bench_serving_metrics(stdout_stderr: str) -> Dict[str, Any]:
    obj = _extract_prefixed_json_line(stdout_stderr, "[run_bench_serving] RESULT_JSON")
    if not obj:
        return {"parse_error": "no_bench_serving_json_found"}

    latency_sec: Optional[float] = None
    try:
        mean_e2e_latency_ms = obj.get("mean_e2e_latency_ms")
        if mean_e2e_latency_ms is not None:
            latency_sec = float(mean_e2e_latency_ms) / 1000.0
    except Exception:
        latency_sec = None

    return {
        "summary": obj,
        "tps": obj.get("output_throughput"),
        "qps": obj.get("request_throughput"),
        "request_throughput": obj.get("request_throughput"),
        "input_throughput": obj.get("input_throughput"),
        "output_throughput": obj.get("output_throughput"),
        "latency_sec": latency_sec,
        "count": obj.get("completed"),
        "num_batches": obj.get("max_concurrency"),
        "mean_e2e_latency_ms": obj.get("mean_e2e_latency_ms"),
        "median_e2e_latency_ms": obj.get("median_e2e_latency_ms"),
        "p99_e2e_latency_ms": obj.get("p99_e2e_latency_ms"),
        "mean_ttft_ms": obj.get("mean_ttft_ms"),
        "median_ttft_ms": obj.get("median_ttft_ms"),
        "p99_ttft_ms": obj.get("p99_ttft_ms"),
        "mean_tpot_ms": obj.get("mean_tpot_ms"),
        "median_tpot_ms": obj.get("median_tpot_ms"),
        "p99_tpot_ms": obj.get("p99_tpot_ms"),
        "max_concurrency": obj.get("max_concurrency"),
    }


def _parse_mteb_metrics(*, output_folder: Path, tasks: List[str], model_id: str, backend: str) -> Dict[str, Any]:
    """Read scripts/embedding/mteb/results/<task>.json and extract run entry."""

    results: Dict[str, Any] = {
        "tasks": tasks,
        "model_id": model_id,
        "backend": backend,
        "per_task": {},
    }

    for task in tasks:
        summary_path = output_folder / "results" / f"{task}.json"
        if not summary_path.exists():
            results["per_task"][task] = {"parse_error": f"missing:{summary_path}"}
            continue

        try:
            data = _load_json(summary_path)
        except Exception as e:
            results["per_task"][task] = {"parse_error": f"invalid_json:{e}"}
            continue

        run: Optional[Dict[str, Any]] = None
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            for r in data["runs"]:
                if not isinstance(r, dict):
                    continue
                if str(r.get("model_name")) == str(model_id) and str(r.get("backend")) == str(backend):
                    run = r
                    break

        if not run:
            results["per_task"][task] = {"parse_error": "no_matching_run"}
            continue

        embedding_stats = run.get("embedding_stats") if isinstance(run, dict) else None
        tps = None
        if isinstance(embedding_stats, dict):
            tps = embedding_stats.get("tps_texts_per_s")

        results["per_task"][task] = {
            "evaluation_time_sec": run.get("evaluation_time"),
            "tps_texts_per_s": tps,
            "embedding_stats": embedding_stats,
            "scores": run.get("scores"),
        }

    # If only one task, lift common fields for CSV convenience
    if len(tasks) == 1:
        t = tasks[0]
        one = results["per_task"].get(t) or {}
        if isinstance(one, dict):
            results["latency_sec"] = one.get("evaluation_time_sec")
            results["tps"] = one.get("tps_texts_per_s")

    return results


def _infer_token_len(*, script: str, env: Dict[str, Any]) -> str:
    """Infer the 'token length' style parameter for reporting.

    - MTEB uses MAX_LENGTH as the truncation length.
    - run_fix_token_len uses SYNTHETIC_TOKEN_LEN (or SYNTHETIC_INPUT_LEN in input_len mode).
    - Other scripts may leave it blank.
    """

    try:
        if script == "run_mteb":
            v = env.get("MAX_LENGTH")
            # Keep consistent with scripts/embedding/mteb/run_mteb.sh and run_mteb.py defaults.
            return str(v) if v not in (None, "") else "512"

        if script == "run_fix_image_size":
            v = env.get("IMAGE_SIZE")
            return str(v) if v not in (None, "") else ""

        if script == "run_fix_token_len":
            mode = str(env.get("MODE") or "").strip().lower()
            if mode in {"token_len", "tokens", "token", "tok"}:
                v = env.get("SYNTHETIC_TOKEN_LEN")
            else:
                v = env.get("SYNTHETIC_INPUT_LEN")
            return str(v) if v not in (None, "") else ""

        if script == "run_bench_serving":
            for key in ["RANDOM_INPUT_LEN", "RANDOM_INPUT", "INPUT_LEN"]:
                v = env.get(key)
                if v not in (None, ""):
                    return str(v)
            return ""

        # Generic fallback
        v = env.get("MAX_LENGTH")
        return str(v) if v not in (None, "") else ""
    except Exception:
        return ""


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _append_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _write_jsonl(path: Path, recs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _resolve_script(script_aliases: Dict[str, str], key: str) -> Path:
    if key not in script_aliases:
        raise SystemExit(f"Unknown script alias: {key}. Known: {sorted(script_aliases.keys())}")
    p = (REPO_ROOT / script_aliases[key]).resolve()
    if not p.exists():
        raise SystemExit(f"Script not found: {p}")
    return p


def _normalize_job_args(*, script: str, raw_args: Any, job_name: str) -> List[str]:
    """Normalize config args into a CLI argv list.

    Supported config formats:
      - args: ["512"]  (legacy positional list)
      - args: {"token_len": 512} (named args; converted to positional for known scripts)
    """

    if raw_args is None:
        return []

    # Legacy list-of-positional
    if isinstance(raw_args, list):
        return [str(a) for a in raw_args]

    # Named args object
    if isinstance(raw_args, dict):
        # Script-specific mapping (make the config self-describing).
        if script == "run_fix_token_len":
            # run_fix_token_len.sh <TOKEN_LEN>
            for k in ("token_len", "input_len", "seq_len", "length", "n_tokens"):
                if k in raw_args:
                    return [str(raw_args[k])]
            # Fallback: first value in insertion order.
            if raw_args:
                return [str(next(iter(raw_args.values())))]
            return []

        if script == "run_mteb":
            # run_mteb.sh <TASK>
            if "task" in raw_args:
                return [str(raw_args["task"])]
            if "tasks" in raw_args:
                # Keep as a single argument; env TASKS is the preferred multi-task interface.
                return [str(raw_args["tasks"])]
            if raw_args:
                return [str(next(iter(raw_args.values())))]
            return []

        # Generic: keep insertion order as positional.
        return [str(v) for v in raw_args.values()]

    raise SystemExit(
        f"Job {job_name} args must be a list or an object (got {type(raw_args).__name__})"
    )


def _build_jobs(cfg: Dict[str, Any]) -> Tuple[Path, Dict[str, str], List[Job]]:
    defaults = cfg.get("defaults") or {}
    default_env = {str(k): str(v) for k, v in (defaults.get("env") or {}).items()}
    default_warmup_runs_raw = defaults.get("warmup_runs", 0)
    try:
        default_warmup_runs = max(0, int(default_warmup_runs_raw))
    except Exception:
        raise SystemExit("defaults.warmup_runs must be an integer >= 0")

    default_repeats = _parse_int_ge(defaults.get("repeats"), default=1, min_value=1, field_name="defaults.repeats")
    default_repeat_threshold_sec: Optional[float] = None
    rts = defaults.get("repeat_threshold_sec")
    if rts is not None and str(rts).strip() != "":
        try:
            v = float(rts)
            if v > 0:
                default_repeat_threshold_sec = float(v)
        except Exception:
            raise SystemExit("defaults.repeat_threshold_sec must be a number > 0")
    default_repeat_max_repeats = _parse_int_ge(
        defaults.get("repeat_max_repeats"), default=100, min_value=1, field_name="defaults.repeat_max_repeats"
    )

    default_restart_servers = _parse_bool(defaults.get("restart_servers"), default=False)
    default_stop_server_after_job = _parse_bool(defaults.get("stop_server_after_job"), default=False)
    result_dir = Path(defaults.get("result_dir") or "scripts/auto-test/embedding/result")
    timeout_sec = defaults.get("timeout_sec")
    timeout_sec = float(timeout_sec) if timeout_sec is not None else None

    script_aliases = {str(k): str(v) for k, v in (cfg.get("script_aliases") or {}).items()}

    jobs_cfg = cfg.get("jobs")
    if not isinstance(jobs_cfg, list) or not jobs_cfg:
        raise SystemExit("config.json must contain a non-empty jobs list")

    jobs: List[Job] = []
    for j in jobs_cfg:
        if not isinstance(j, dict):
            continue
        name = str(j.get("name") or "")
        if not name:
            raise SystemExit("Each job must have a name")
        script = str(j.get("script") or "")
        if not script:
            raise SystemExit(f"Job {name} missing script")
        args = _normalize_job_args(script=script, raw_args=j.get("args"), job_name=name)
        env = _merge_env(default_env, {str(k): str(v) for k, v in (j.get("env") or {}).items()})
        job_timeout = j.get("timeout_sec")
        job_timeout_sec = float(job_timeout) if job_timeout is not None else timeout_sec
        warmup_runs_raw = j.get("warmup_runs", default_warmup_runs)
        try:
            warmup_runs = max(0, int(warmup_runs_raw))
        except Exception:
            raise SystemExit(f"jobs[{name}].warmup_runs must be an integer >= 0")

        repeats = _parse_int_ge(j.get("repeats", default_repeats), default=default_repeats, min_value=1, field_name=f"jobs[{name}].repeats")

        repeat_threshold_sec: Optional[float] = default_repeat_threshold_sec
        rts_j = j.get("repeat_threshold_sec")
        if rts_j is not None and str(rts_j).strip() != "":
            try:
                v = float(rts_j)
            except Exception:
                raise SystemExit(f"jobs[{name}].repeat_threshold_sec must be a number > 0")
            if v > 0:
                repeat_threshold_sec = float(v)
            else:
                repeat_threshold_sec = None

        repeat_max_repeats = _parse_int_ge(
            j.get("repeat_max_repeats", default_repeat_max_repeats),
            default=default_repeat_max_repeats,
            min_value=1,
            field_name=f"jobs[{name}].repeat_max_repeats",
        )

        restart_servers = _parse_bool(j.get("restart_servers"), default=default_restart_servers)
        stop_server_after_job = _parse_bool(j.get("stop_server_after_job"), default=default_stop_server_after_job)
        emon_enable = _parse_bool(j.get("emon_enable"), default=False)

        jobs.append(
            Job(
                name=name,
                script=script,
                args=args,
                env=env,
                timeout_sec=job_timeout_sec,
                warmup_runs=warmup_runs,
                repeats=repeats,
                repeat_threshold_sec=repeat_threshold_sec,
                repeat_max_repeats=repeat_max_repeats,
                restart_servers=restart_servers,
                stop_server_after_job=stop_server_after_job,
                emon_enable=emon_enable,
            )
        )

    return result_dir, script_aliases, jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-test embedding scripts via a JSON config")
    ap.add_argument(
        "--config",
        default=str(REPO_ROOT / "scripts/auto-test/embedding/config_yahoo.json"),
        help="Path to config JSON",
    )
    ap.add_argument("--only", action="append", default=[], help="Only run jobs with this exact name (repeatable)")
    ap.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip jobs with this exact name (repeatable)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands but do not execute")
    ap.add_argument(
        "--tee",
        action="store_true",
        help="Stream server/job logs to the terminal while also writing log files",
    )
    ap.add_argument(
        "--restart-servers",
        action="store_true",
        help="If a local server is already listening on the configured port, terminate it and start a fresh one",
    )
    ap.add_argument(
        "--stop-servers-after-job",
        action="store_true",
        help="Shutdown any server started by the runner after each job completes (useful when jobs use different MODEL_DIR/SERVED_MODEL_NAME)",
    )
    ap.add_argument(
        "--reparse-run-id",
        default="",
        help="Re-parse existing logs/metrics for a prior run_id and rewrite summary_<run_id>.jsonl/.csv",
    )
    args = ap.parse_args()

    # If the scale-test wrapper provided a CPU list, enforce it here as well.
    # This makes CPU pinning resilient even when systemd-run properties are
    # accepted but not enforced by the host.
    _apply_self_affinity_from_env(tee=bool(args.tee))

    cfg_path = Path(args.config)
    cfg = _load_json(cfg_path)

    result_dir, script_aliases, jobs = _build_jobs(cfg)
    servers = _parse_servers(cfg)
    result_dir = (REPO_ROOT / result_dir).resolve() if not result_dir.is_absolute() else result_dir
    _ensure_dir(result_dir)

    # Re-parse mode: regenerate metrics/summary from existing log files.
    if str(args.reparse_run_id or "").strip():
        run_id = str(args.reparse_run_id).strip()
        # New layout prefers per-run directory; keep backward-compatible fallback.
        run_dir = result_dir / run_id
        search_dir = run_dir if run_dir.exists() else result_dir
        metrics_files = sorted(search_dir.glob(f"{run_id}_*.metrics.json"))
        if not metrics_files:
            raise SystemExit(f"No metrics files found for run_id={run_id} in {result_dir}")

        csv_fields = [
            "job_name",
            "script",
            "exit_code",
            "repeats",
            "backend",
            "model",
            "model_id",
            "numactl_cores",
            "numactl_cpunodebind",
            "numactl_membind",
            "token_len",
            "tps",
            "tps_std",
            "qps",
            "qps_std",
            "latency_sec",
            "latency_sec_std",
            "mean_ttft_ms",
            "mean_ttft_ms_std",
            "mean_tpot_ms",
            "mean_tpot_ms_std",
            "avg_batch_time_sec",
            "avg_batch_time_sec_std",
            "count",
            "num_batches",
            "started_at_utc",
            "ended_at_utc",
            "log_path",
            "metrics_path",
        ]

        recs: List[Dict[str, Any]] = []
        csv_rows: List[Dict[str, Any]] = []
        for mp in metrics_files:
            rec = _load_json(mp)
            if not isinstance(rec, dict):
                continue

            script = str(rec.get("script") or "")
            backend = _norm_backend(str(rec.get("backend") or (rec.get("server") or {}).get("backend") or ""))
            model = str(rec.get("model") or "")
            model_id = str(rec.get("model_id") or "")

            env = rec.get("env") or {}
            server_info = rec.get("server") or {}

            token_len = _infer_token_len(script=script, env=env if isinstance(env, dict) else {})

            # Prefer the effective server numactl recorded at runtime; else recompute from env + config defaults.
            numactl_cores = server_info.get("numactl_cores")
            numactl_cpunodebind = server_info.get("numactl_cpunodebind")
            numactl_membind = server_info.get("numactl_membind")

            if (not (numactl_cores or numactl_cpunodebind or numactl_membind)) and backend in servers:
                # Only fill from server defaults if this script/backend likely uses a server.
                try:
                    eff_cores, eff_cpunodebind, eff_membind = _effective_numactl_for_server(
                        spec=servers[backend],
                        job_env={str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {},
                    )
                    numactl_cores = numactl_cores or eff_cores
                    numactl_cpunodebind = numactl_cpunodebind or eff_cpunodebind
                    numactl_membind = numactl_membind or eff_membind
                except Exception:
                    pass

            log_path = Path(str(rec.get("log_path") or ""))
            # If we have per-repeat logs recorded, re-aggregate metrics from those logs.
            repeat_log_paths_raw = rec.get("repeat_log_paths")
            repeat_log_paths: List[Path] = []
            if isinstance(repeat_log_paths_raw, list):
                for x in repeat_log_paths_raw:
                    try:
                        p = Path(str(x))
                        repeat_log_paths.append(p)
                    except Exception:
                        continue

            per_repeat: List[Dict[str, Any]] = []
            if repeat_log_paths:
                for i, lp in enumerate(repeat_log_paths, start=1):
                    if not lp.exists():
                        per_repeat.append(
                            {
                                "rep": int(i),
                                "exit_code": int(1),
                                "log_path": str(lp),
                                "metrics": {"parse_error": f"missing_log:{lp}"},
                            }
                        )
                        continue
                    try:
                        text_i = lp.read_text(encoding="utf-8")
                    except Exception as e:
                        per_repeat.append(
                            {
                                "rep": int(i),
                                "exit_code": int(1),
                                "log_path": str(lp),
                                "metrics": {"parse_error": f"read_log_error:{e}"},
                            }
                        )
                        continue

                    # We don't have per-repeat exit codes in logs; assume OK if parsable.
                    if script in ("run_embedding_yahoo", "run_fix_token_len", "run_fix_image_size"):
                        metrics_i = _parse_embedding_metrics(text_i)
                    elif script == "run_bench_serving":
                        metrics_i = _parse_bench_serving_metrics(text_i)
                    elif script == "run_mteb":
                        tasks = [t.strip() for t in str(env.get("TASKS") or "").split(",") if t.strip()]
                        if not tasks:
                            cmd = rec.get("cmd") or []
                            if isinstance(cmd, list) and len(cmd) >= 3:
                                tasks = [str(cmd[2]).strip()]
                        output_folder = Path(str(env.get("OUTPUT_FOLDER") or "scripts/embedding/mteb"))
                        output_folder = (
                            (REPO_ROOT / output_folder).resolve() if not output_folder.is_absolute() else output_folder
                        )
                        metrics_i = _parse_mteb_metrics(
                            output_folder=output_folder, tasks=tasks, model_id=model_id, backend=backend
                        )
                    else:
                        metrics_i = {"parse_error": f"no_parser_for_script:{script}"}

                    exit_code_i = 0 if not str(metrics_i.get("parse_error") or "").strip() else 1
                    per_repeat.append(
                        {
                            "rep": int(i),
                            "exit_code": int(exit_code_i),
                            "log_path": str(lp),
                            "metrics": metrics_i,
                        }
                    )

                metrics = _aggregate_repeat_metrics(script=script, per_repeat=per_repeat)
            else:
                metrics = {}
                if script in ("run_embedding_yahoo", "run_fix_token_len", "run_fix_image_size"):
                    if log_path.exists():
                        text = log_path.read_text(encoding="utf-8")
                        metrics = _parse_embedding_metrics(text)
                    else:
                        metrics = {"parse_error": f"missing_log:{log_path}"}
                elif script == "run_bench_serving":
                    if log_path.exists():
                        text = log_path.read_text(encoding="utf-8")
                        metrics = _parse_bench_serving_metrics(text)
                    else:
                        metrics = {"parse_error": f"missing_log:{log_path}"}
                elif script == "run_mteb":
                    tasks = [t.strip() for t in str(env.get("TASKS") or "").split(",") if t.strip()]
                    if not tasks:
                        cmd = rec.get("cmd") or []
                        if isinstance(cmd, list) and len(cmd) >= 3:
                            tasks = [str(cmd[2]).strip()]
                    output_folder = Path(str(env.get("OUTPUT_FOLDER") or "scripts/embedding/mteb"))
                    output_folder = (
                        (REPO_ROOT / output_folder).resolve() if not output_folder.is_absolute() else output_folder
                    )
                    metrics = _parse_mteb_metrics(
                        output_folder=output_folder, tasks=tasks, model_id=model_id, backend=backend
                    )
                else:
                    metrics = {"parse_error": f"no_parser_for_script:{script}"}

            rec["metrics"] = metrics
            rec["tps"] = metrics.get("tps")
            rec["latency_sec"] = metrics.get("latency_sec")
            rec["tps_std"] = metrics.get("tps_std")
            rec["latency_sec_std"] = metrics.get("latency_sec_std")
            rec["qps"] = metrics.get("qps") or metrics.get("request_throughput")
            rec["qps_std"] = metrics.get("qps_std") or metrics.get("request_throughput_std")
            rec["mean_ttft_ms"] = metrics.get("mean_ttft_ms")
            rec["mean_ttft_ms_std"] = metrics.get("mean_ttft_ms_std")
            rec["mean_tpot_ms"] = metrics.get("mean_tpot_ms")
            rec["mean_tpot_ms_std"] = metrics.get("mean_tpot_ms_std")
            _write_json(mp, rec)

            recs.append(rec)
            csv_rows.append(
                {
                    "job_name": rec.get("job_name"),
                    "script": script,
                    "exit_code": rec.get("exit_code"),
                    "repeats": rec.get("repeats") or metrics.get("repeats") or "",
                    "backend": backend,
                    "model": model,
                    "model_id": model_id,
                    "numactl_cores": numactl_cores,
                    "numactl_cpunodebind": numactl_cpunodebind,
                    "numactl_membind": numactl_membind,
                    "token_len": token_len,
                    "tps": rec.get("tps"),
                    "tps_std": rec.get("tps_std") or metrics.get("tps_std") or "",
                    "qps": rec.get("qps") or metrics.get("qps") or metrics.get("request_throughput") or "",
                    "qps_std": rec.get("qps_std") or metrics.get("qps_std") or metrics.get("request_throughput_std") or "",
                    "latency_sec": rec.get("latency_sec"),
                    "latency_sec_std": rec.get("latency_sec_std") or metrics.get("latency_sec_std") or "",
                    "mean_ttft_ms": rec.get("mean_ttft_ms") or metrics.get("mean_ttft_ms") or "",
                    "mean_ttft_ms_std": rec.get("mean_ttft_ms_std") or metrics.get("mean_ttft_ms_std") or "",
                    "mean_tpot_ms": rec.get("mean_tpot_ms") or metrics.get("mean_tpot_ms") or "",
                    "mean_tpot_ms_std": rec.get("mean_tpot_ms_std") or metrics.get("mean_tpot_ms_std") or "",
                    "avg_batch_time_sec": metrics.get("avg_batch_time_sec"),
                    "avg_batch_time_sec_std": metrics.get("avg_batch_time_sec_std"),
                    "count": metrics.get("count"),
                    "num_batches": metrics.get("num_batches"),
                    "started_at_utc": rec.get("started_at_utc"),
                    "ended_at_utc": rec.get("ended_at_utc"),
                    "log_path": rec.get("log_path"),
                    "metrics_path": str(mp),
                }
            )

        summary_jsonl = result_dir / f"summary_{run_id}.jsonl"
        summary_csv = result_dir / f"summary_{run_id}.csv"
        # Keep CSV at root; write JSONL into the per-run directory.
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_jsonl = run_dir / f"summary_{run_id}.jsonl"
        _write_jsonl(summary_jsonl, recs)
        _write_csv(summary_csv, csv_rows, csv_fields)
        print(f"Rewrote: {summary_jsonl}")
        print(f"Rewrote: {summary_csv}")
        return 0

    run_id = _utc_now_compact()
    run_dir = result_dir / run_id
    _ensure_dir(run_dir)
    # Keep CSV at root; put everything else under run_dir.
    summary_jsonl = run_dir / f"summary_{run_id}.jsonl"
    summary_csv = result_dir / f"summary_{run_id}.csv"

    only_set = set(args.only or [])
    if only_set:
        jobs = [j for j in jobs if j.name in only_set]

    skip_set = set(args.skip or [])
    if skip_set:
        jobs = [j for j in jobs if j.name not in skip_set]

    if not jobs:
        print("No jobs selected.", file=sys.stderr)
        return 2

    csv_fields = [
        "job_name",
        "script",
        "exit_code",
        "repeats",
        "backend",
        "model",
        "model_id",
        "numactl_cores",
        "numactl_cpunodebind",
        "numactl_membind",
        "token_len",
        "tps",
        "tps_std",
        "qps",
        "qps_std",
        "latency_sec",
        "latency_sec_std",
        "mean_ttft_ms",
        "mean_ttft_ms_std",
        "mean_tpot_ms",
        "mean_tpot_ms_std",
        "avg_batch_time_sec",
        "avg_batch_time_sec_std",
        "count",
        "num_batches",
        "started_at_utc",
        "ended_at_utc",
        "log_path",
        "metrics_path",
    ]

    csv_rows: List[Dict[str, Any]] = []

    running_servers: Dict[str, RunningServer] = {}

    try:
        for idx, job in enumerate(jobs, start=1):
            script_path = _resolve_script(script_aliases, job.script)
            started = dt.datetime.now(tz=dt.timezone.utc)
            started_s = started.isoformat()

            safe = _sanitize_filename(job.name)
            # Keep a stable "job log" path for backward compatibility, but store per-repeat
            # logs separately (rep_XX). The job log will point to the last repeat.
            log_path = run_dir / f"{run_id}_{idx:03d}_{safe}.log"
            metrics_path = run_dir / f"{run_id}_{idx:03d}_{safe}.metrics.json"

            cmd = ["bash", str(script_path)] + list(job.args)
            env = os.environ.copy()
            env.update(job.env)

            # Optional NUMA binding for the benchmark/job process.
            job_numactl_cores, job_numactl_cpunodebind, job_numactl_membind = _effective_numactl_for_job(job_env=job.env)
            cmd_job = _maybe_wrap_cmd_with_numactl(
                cmd=cmd,
                numactl_cores=job_numactl_cores,
                numactl_cpunodebind=job_numactl_cpunodebind,
                numactl_membind=job_numactl_membind,
            )

            backend_raw = job.env.get("BACKEND") or ""
            backend = _norm_backend(backend_raw)
            base_url = _norm_base_url(job.env.get("BASE_URL") or "")

            server_info: Dict[str, Any] = {}
            job_server_key: str = ""
            if backend in servers and base_url:
                effective_restart_servers = bool(args.restart_servers) or bool(job.restart_servers)
                rs = _ensure_server(
                    spec=servers[backend],
                    job=job,
                    base_url=base_url,
                    result_dir=run_dir,
                    run_id=run_id,
                    idx=idx,
                    dry_run=args.dry_run,
                    tee=args.tee,
                    restart_servers=effective_restart_servers,
                    running=running_servers,
                )
                if rs is not None:
                    job_server_key = _server_key(servers[backend], base_url)
                    server_info = {
                        "backend": backend,
                        "base_url": base_url,
                        "started_by_runner": rs.started_by_runner,
                        "server_log_path": str(rs.log_path),
                        "numactl_cores": rs.numactl_cores,
                        "numactl_cpunodebind": rs.numactl_cpunodebind,
                        "numactl_membind": rs.numactl_membind,
                    }

            if args.dry_run:
                if server_info:
                    print(f"[dry-run] {job.name}: (server ready) {' '.join(cmd_job)}")
                else:
                    print(f"[dry-run] {job.name}: {' '.join(cmd_job)}")
                continue

            # Warmup runs: same command, but do NOT parse metrics or append to summary.
            for w in range(int(job.warmup_runs or 0)):
                warmup_log_path = run_dir / f"{run_id}_{idx:03d}_{safe}.warmup_{w+1:02d}.log"
                rc_w, _, _ = _run_job_streaming(
                    cmd=cmd_job,
                    cwd=REPO_ROOT,
                    env=env,
                    log_path=warmup_log_path,
                    tee=args.tee,
                    prefix=f"[{job.name} warmup {w+1}/{job.warmup_runs}] ",
                    timeout_sec=job.timeout_sec,
                )
                if rc_w != 0:
                    raise SystemExit(
                        f"Warmup failed for job={job.name} (exit_code={rc_w}). See {warmup_log_path}"
                    )

            emon_session: Optional[EmonSession] = None
            emon_output_path: Optional[Path] = None
            if bool(job.emon_enable):
                emon_dir = run_dir / f"{run_id}_{idx:03d}_{safe}.emon"
                emon_output_path = emon_dir / "emon.dat"
                emon_session = _start_emon_session(
                    output_path=emon_output_path,
                    tee=bool(args.tee),
                    prefix=f"[{job.name}] ",
                )

            try:
                per_repeat: List[Dict[str, Any]] = []
                repeat_log_paths: List[str] = []
                total_wall_time_sec = 0.0

                repeats_config = max(1, int(job.repeats))
                repeats_effective = repeats_config
                repeats_computed: Optional[int] = None
                repeats_capped = False

                first_nonzero_exit_code: Optional[int] = None
                any_success = False
                last_combined_output = ""

                def _run_one_repeat(rep: int, *, rep_total_hint: str) -> None:
                    nonlocal total_wall_time_sec, last_combined_output, any_success, first_nonzero_exit_code

                    rep_log_path = run_dir / f"{run_id}_{idx:03d}_{safe}.rep_{rep:02d}.log"
                    repeat_log_paths.append(str(rep_log_path))

                    rc_i, out_i, wall_i = _run_job_streaming(
                        cmd=cmd_job,
                        cwd=REPO_ROOT,
                        env=env,
                        log_path=rep_log_path,
                        tee=args.tee,
                        prefix=f"[{job.name} rep {rep}/{rep_total_hint}] ",
                        timeout_sec=job.timeout_sec,
                    )
                    total_wall_time_sec += float(wall_i)
                    last_combined_output = out_i
                    if int(rc_i) == 0:
                        any_success = True
                    elif first_nonzero_exit_code is None:
                        first_nonzero_exit_code = int(rc_i)

                    # Parse per-repeat metrics (best-effort).
                    if job.script in ("run_embedding_yahoo", "run_fix_token_len", "run_fix_image_size"):
                        metrics_i = _parse_embedding_metrics(out_i)
                    elif job.script == "run_bench_serving":
                        metrics_i = _parse_bench_serving_metrics(out_i)
                    elif job.script == "run_mteb":
                        tasks_i: List[str] = []
                        if job.args:
                            tasks_i = [str(job.args[0]).strip()]
                        else:
                            tasks_i = [t.strip() for t in (job.env.get("TASKS") or "").split(",") if t.strip()]
                        output_folder_i = Path(job.env.get("OUTPUT_FOLDER") or "scripts/embedding/mteb")
                        output_folder_i = (
                            (REPO_ROOT / output_folder_i).resolve() if not output_folder_i.is_absolute() else output_folder_i
                        )
                        metrics_i = _parse_mteb_metrics(
                            output_folder=output_folder_i,
                            tasks=tasks_i,
                            model_id=str(job.env.get("MODEL_ID") or ""),
                            backend=backend,
                        )
                    else:
                        metrics_i = {"parse_error": f"no_parser_for_script:{job.script}"}

                    per_repeat.append(
                        {
                            "rep": int(rep),
                            "exit_code": int(rc_i),
                            "log_path": str(rep_log_path),
                            "wall_time_sec": float(wall_i),
                            "metrics": metrics_i,
                        }
                    )

                # Threshold mode: always run first repeat, then decide how many total repeats.
                if job.repeat_threshold_sec is not None and float(job.repeat_threshold_sec) > 0:
                    _run_one_repeat(1, rep_total_hint="auto")

                    wall1 = None
                    if per_repeat:
                        wall1 = _to_float_or_none(per_repeat[0].get("wall_time_sec"))

                    thr = float(job.repeat_threshold_sec)
                    if wall1 is not None and wall1 > 0 and wall1 < thr:
                        repeats_computed = max(1, int(thr / wall1))
                    else:
                        repeats_computed = 1

                    repeats_effective = max(1, int(repeats_computed))
                    if repeats_effective > int(job.repeat_max_repeats):
                        repeats_effective = int(job.repeat_max_repeats)
                        repeats_capped = True

                    # Run remaining repeats.
                    for rep in range(2, repeats_effective + 1):
                        _run_one_repeat(rep, rep_total_hint=str(repeats_effective))
                else:
                    repeats_effective = repeats_config
                    for rep in range(1, repeats_effective + 1):
                        _run_one_repeat(rep, rep_total_hint=str(repeats_effective))

                # For compatibility: write the last repeat output to the legacy log_path.
                # (So existing tooling that expects *.log can still find something.)
                try:
                    log_path.write_text(last_combined_output or "", encoding="utf-8")
                except Exception:
                    pass

                # If at least one repeat succeeds, consider the overall job successful.
                # This keeps scale-test analysis robust to occasional flaky failures.
                exit_code = 0 if any_success else int(first_nonzero_exit_code or 1)
                combined_output = str(last_combined_output)
                wall_time_sec = float(total_wall_time_sec)
            finally:
                if emon_session is not None:
                    _stop_emon_session(emon_session, tee=bool(args.tee), prefix=f"[{job.name}] ")

            ended = dt.datetime.now(tz=dt.timezone.utc)
            ended_s = ended.isoformat()

            base_rec: Dict[str, Any] = {
                "run_id": run_id,
                "job_name": job.name,
                "script": job.script,
                "script_path": str(script_path),
                "cmd": cmd_job,
                "exit_code": exit_code,
                "started_at_utc": started_s,
                "ended_at_utc": ended_s,
                "wall_time_sec": wall_time_sec,
                "env": {k: job.env.get(k) for k in sorted(job.env.keys())},
                "log_path": str(log_path),
                # repeats is the effective repeats actually executed.
                "repeats": int(repeats_effective),
                "repeats_config": int(repeats_config),
                "repeats_computed": int(repeats_computed) if repeats_computed is not None else None,
                "repeats_capped": bool(repeats_capped),
                "repeat_threshold_sec": float(job.repeat_threshold_sec) if job.repeat_threshold_sec is not None else None,
                "repeat_max_repeats": int(job.repeat_max_repeats),
                "repeat_log_paths": repeat_log_paths,
                "server": server_info,
                "job_numactl": {
                    "cores": job_numactl_cores,
                    "cpunodebind": job_numactl_cpunodebind,
                    "membind": job_numactl_membind,
                },
                "emon": {
                    "enabled": bool(job.emon_enable),
                    "output_path": str(emon_output_path) if emon_output_path is not None else "",
                },
            }

            model = job.env.get("MODEL") or ""
            model_id = job.env.get("MODEL_ID") or ""

            # Aggregate per-repeat metrics into a single metrics dict with mean/std.
            metrics = _aggregate_repeat_metrics(script=job.script, per_repeat=per_repeat)

            merged = dict(base_rec)
            merged["metrics"] = metrics
            merged["tps"] = metrics.get("tps")
            merged["latency_sec"] = metrics.get("latency_sec")
            merged["tps_std"] = metrics.get("tps_std")
            merged["latency_sec_std"] = metrics.get("latency_sec_std")
            merged["qps"] = metrics.get("qps") or metrics.get("request_throughput")
            merged["qps_std"] = metrics.get("qps_std") or metrics.get("request_throughput_std")
            merged["mean_ttft_ms"] = metrics.get("mean_ttft_ms")
            merged["mean_ttft_ms_std"] = metrics.get("mean_ttft_ms_std")
            merged["mean_tpot_ms"] = metrics.get("mean_tpot_ms")
            merged["mean_tpot_ms_std"] = metrics.get("mean_tpot_ms_std")
            merged["backend"] = backend
            merged["model"] = model
            merged["model_id"] = model_id

            _write_json(metrics_path, merged)
            _append_jsonl(summary_jsonl, merged)

            csv_rows.append(
                {
                    "job_name": job.name,
                    "script": job.script,
                    "exit_code": exit_code,
                    "repeats": int(repeats_effective),
                    "backend": backend,
                    "model": model,
                    "model_id": model_id,
                    # Keep these columns representing *server* binding.
                    "numactl_cores": server_info.get("numactl_cores"),
                    "numactl_cpunodebind": server_info.get("numactl_cpunodebind"),
                    "numactl_membind": server_info.get("numactl_membind"),
                    "token_len": _infer_token_len(script=job.script, env=job.env),
                    "tps": merged.get("tps"),
                    "tps_std": merged.get("tps_std"),
                    "qps": merged.get("qps"),
                    "qps_std": merged.get("qps_std"),
                    "latency_sec": merged.get("latency_sec"),
                    "latency_sec_std": merged.get("latency_sec_std"),
                    "mean_ttft_ms": merged.get("mean_ttft_ms"),
                    "mean_ttft_ms_std": merged.get("mean_ttft_ms_std"),
                    "mean_tpot_ms": merged.get("mean_tpot_ms"),
                    "mean_tpot_ms_std": merged.get("mean_tpot_ms_std"),
                    "avg_batch_time_sec": metrics.get("avg_batch_time_sec"),
                    "avg_batch_time_sec_std": metrics.get("avg_batch_time_sec_std"),
                    "count": metrics.get("count"),
                    "num_batches": metrics.get("num_batches"),
                    "started_at_utc": started_s,
                    "ended_at_utc": ended_s,
                    "log_path": str(log_path),
                    "metrics_path": str(metrics_path),
                }
            )

            # Optional: stop the server after each job so subsequent jobs can start with different env/model.
            if (bool(args.stop_servers_after_job) or bool(job.stop_server_after_job)) and job_server_key:
                rs = running_servers.get(job_server_key)
                if rs is not None and rs.started_by_runner:
                    if args.tee:
                        print(f"[server:{rs.spec.backend}] stopping after job={job.name}")
                    _teardown_server(rs)
                    running_servers.pop(job_server_key, None)

        if not args.dry_run:
            _append_csv(summary_csv, csv_rows, csv_fields)
            print(f"Wrote: {summary_jsonl}")
            print(f"Wrote: {summary_csv}")
            print(f"Per-run logs/metrics in: {run_dir}")

        return 0
    finally:
        if not args.dry_run:
            _teardown_servers(running_servers)


if __name__ == "__main__":
    raise SystemExit(main())
