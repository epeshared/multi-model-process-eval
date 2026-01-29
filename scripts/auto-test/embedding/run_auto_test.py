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


@dataclass(frozen=True)
class Job:
    name: str
    script: str
    args: List[str]
    env: Dict[str, str]
    timeout_sec: Optional[float]
    warmup_runs: int
    restart_servers: bool


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
) -> bool:
    base_url = _norm_base_url(base_url)
    if not base_url:
        return False

    eps = endpoints or ["POST /v1/embeddings", "/v1/models", "/health"]
    deadline = time.time() + max(1.0, float(timeout_sec))
    while time.time() < deadline:
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
        if restart_servers:
            # If it is still ready after a restart attempt, treat as external and proceed.
            if tee:
                print(f"[server:{spec.backend}] still running after restart attempt; treating as external")
        else:
            if tee:
                print(f"[server:{spec.backend}] already running at {base_url} (external); will not start")
            rs = RunningServer(
                spec=spec,
                base_url=base_url,
                proc=None,
                started_by_runner=False,
                log_path=result_dir / f"{run_id}_{idx:03d}_server_{_sanitize_filename(spec.backend)}.external.log",
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

    log_path = result_dir / f"{run_id}_{idx:03d}_server_{_sanitize_filename(spec.backend)}.log"

    cmd = ["bash", str(script_path)] + list(spec.args)

    # Optional NUMA binding for server process.
    # Priority: per-job env overrides > servers.<backend>.numactl defaults.
    numactl_cores = (job.env.get("NUMACTL_CORES") or "").strip() or (spec.numactl_cores or "").strip()
    numactl_cpunodebind = (job.env.get("NUMACTL_CPUNODEBIND") or "").strip() or (
        spec.numactl_cpunodebind or ""
    ).strip()
    numactl_membind = (job.env.get("NUMACTL_MEMBIND") or "").strip() or (spec.numactl_membind or "").strip()

    if numactl_cores or numactl_cpunodebind or numactl_membind:
        numactl_bin = shutil.which("numactl")
        if not numactl_bin:
            warn_bits = []
            if numactl_cores:
                warn_bits.append(f"NUMACTL_CORES={numactl_cores}")
            if numactl_cpunodebind:
                warn_bits.append(f"NUMACTL_CPUNODEBIND={numactl_cpunodebind}")
            if numactl_membind:
                warn_bits.append(f"NUMACTL_MEMBIND={numactl_membind}")
            print(f"[warn] numactl requested ({', '.join(warn_bits)}) but numactl not found; starting without binding")
        else:
            prefix: List[str] = [numactl_bin]
            if numactl_cores:
                prefix += ["-C", numactl_cores]
            if numactl_cpunodebind:
                prefix += ["--cpunodebind", numactl_cpunodebind]
            if numactl_membind:
                prefix += ["--membind", numactl_membind]
            prefix += ["--"]
            cmd = prefix + cmd

    if dry_run:
        print(f"[dry-run] start-server backend={spec.backend} base_url={base_url}: {' '.join(cmd)}")
        rs = RunningServer(spec=spec, base_url=base_url, proc=None, started_by_runner=True, log_path=log_path)
        running[key] = rs
        return rs

    if tee:
        print(f"[server:{spec.backend}] starting (log: {log_path})")

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
    )
    if not ok:
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
    )
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


def _build_jobs(cfg: Dict[str, Any]) -> Tuple[Path, Dict[str, str], List[Job]]:
    defaults = cfg.get("defaults") or {}
    default_env = {str(k): str(v) for k, v in (defaults.get("env") or {}).items()}
    default_warmup_runs_raw = defaults.get("warmup_runs", 0)
    try:
        default_warmup_runs = max(0, int(default_warmup_runs_raw))
    except Exception:
        raise SystemExit("defaults.warmup_runs must be an integer >= 0")

    default_restart_servers = _parse_bool(defaults.get("restart_servers"), default=False)
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
        args = j.get("args") or []
        if not isinstance(args, list):
            raise SystemExit(f"Job {name} args must be a list")
        env = _merge_env(default_env, {str(k): str(v) for k, v in (j.get("env") or {}).items()})
        job_timeout = j.get("timeout_sec")
        job_timeout_sec = float(job_timeout) if job_timeout is not None else timeout_sec
        warmup_runs_raw = j.get("warmup_runs", default_warmup_runs)
        try:
            warmup_runs = max(0, int(warmup_runs_raw))
        except Exception:
            raise SystemExit(f"jobs[{name}].warmup_runs must be an integer >= 0")

        restart_servers = _parse_bool(j.get("restart_servers"), default=default_restart_servers)

        jobs.append(
            Job(
                name=name,
                script=script,
                args=[str(a) for a in args],
                env=env,
                timeout_sec=job_timeout_sec,
                warmup_runs=warmup_runs,
                restart_servers=restart_servers,
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
        "--reparse-run-id",
        default="",
        help="Re-parse existing logs/metrics for a prior run_id and rewrite summary_<run_id>.jsonl/.csv",
    )
    args = ap.parse_args()

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
            "run_id",
            "job_name",
            "script",
            "exit_code",
            "backend",
            "model",
            "model_id",
            "tps",
            "latency_sec",
            "avg_batch_time_sec",
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

            log_path = Path(str(rec.get("log_path") or ""))
            metrics: Dict[str, Any]
            if script in ("run_embedding_yahoo", "run_fix_token_len"):
                if log_path.exists():
                    text = log_path.read_text(encoding="utf-8")
                    metrics = _parse_embedding_metrics(text)
                else:
                    metrics = {"parse_error": f"missing_log:{log_path}"}
            elif script == "run_mteb":
                env = rec.get("env") or {}
                tasks = [t.strip() for t in str(env.get("TASKS") or "").split(",") if t.strip()]
                if not tasks:
                    # Fallback: try to infer from cmd[2] (bash script arg)
                    cmd = rec.get("cmd") or []
                    if isinstance(cmd, list) and len(cmd) >= 3:
                        tasks = [str(cmd[2]).strip()]
                output_folder = Path(str(env.get("OUTPUT_FOLDER") or "scripts/embedding/mteb"))
                output_folder = (REPO_ROOT / output_folder).resolve() if not output_folder.is_absolute() else output_folder
                metrics = _parse_mteb_metrics(output_folder=output_folder, tasks=tasks, model_id=model_id, backend=backend)
            else:
                metrics = {"parse_error": f"no_parser_for_script:{script}"}

            rec["metrics"] = metrics
            rec["tps"] = metrics.get("tps")
            rec["latency_sec"] = metrics.get("latency_sec")
            _write_json(mp, rec)

            recs.append(rec)
            csv_rows.append(
                {
                    "run_id": rec.get("run_id"),
                    "job_name": rec.get("job_name"),
                    "script": script,
                    "exit_code": rec.get("exit_code"),
                    "backend": backend,
                    "model": model,
                    "model_id": model_id,
                    "tps": rec.get("tps"),
                    "latency_sec": rec.get("latency_sec"),
                    "avg_batch_time_sec": metrics.get("avg_batch_time_sec"),
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

    if not jobs:
        print("No jobs selected.", file=sys.stderr)
        return 2

    csv_fields = [
        "run_id",
        "job_name",
        "script",
        "exit_code",
        "backend",
        "model",
        "model_id",
        "tps",
        "latency_sec",
        "avg_batch_time_sec",
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
            log_path = run_dir / f"{run_id}_{idx:03d}_{safe}.log"
            metrics_path = run_dir / f"{run_id}_{idx:03d}_{safe}.metrics.json"

            cmd = ["bash", str(script_path)] + list(job.args)
            env = os.environ.copy()
            env.update(job.env)

            backend_raw = job.env.get("BACKEND") or ""
            backend = _norm_backend(backend_raw)
            base_url = _norm_base_url(job.env.get("BASE_URL") or "")

            server_info: Dict[str, Any] = {}
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
                    server_info = {
                        "backend": backend,
                        "base_url": base_url,
                        "started_by_runner": rs.started_by_runner,
                        "server_log_path": str(rs.log_path),
                    }

            if args.dry_run:
                if server_info:
                    print(f"[dry-run] {job.name}: (server ready) {' '.join(cmd)}")
                else:
                    print(f"[dry-run] {job.name}: {' '.join(cmd)}")
                continue

            # Warmup runs: same command, but do NOT parse metrics or append to summary.
            for w in range(int(job.warmup_runs or 0)):
                warmup_log_path = run_dir / f"{run_id}_{idx:03d}_{safe}.warmup_{w+1:02d}.log"
                rc_w, _, _ = _run_job_streaming(
                    cmd=cmd,
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

            exit_code, combined_output, wall_time_sec = _run_job_streaming(
                cmd=cmd,
                cwd=REPO_ROOT,
                env=env,
                log_path=log_path,
                tee=args.tee,
                prefix=f"[{job.name}] ",
                timeout_sec=job.timeout_sec,
            )

            ended = dt.datetime.now(tz=dt.timezone.utc)
            ended_s = ended.isoformat()

            base_rec: Dict[str, Any] = {
                "run_id": run_id,
                "job_name": job.name,
                "script": job.script,
                "script_path": str(script_path),
                "cmd": cmd,
                "exit_code": exit_code,
                "started_at_utc": started_s,
                "ended_at_utc": ended_s,
                "wall_time_sec": wall_time_sec,
                "env": {k: job.env.get(k) for k in sorted(job.env.keys())},
                "log_path": str(log_path),
                "server": server_info,
            }

            model = job.env.get("MODEL") or ""
            model_id = job.env.get("MODEL_ID") or ""

            metrics: Dict[str, Any]
            if job.script in ("run_embedding_yahoo", "run_fix_token_len"):
                metrics = _parse_embedding_metrics(combined_output)
            elif job.script == "run_mteb":
                tasks: List[str] = []
                if job.args:
                    tasks = [str(job.args[0]).strip()]
                else:
                    tasks = [t.strip() for t in (job.env.get("TASKS") or "").split(",") if t.strip()]
                output_folder = Path(job.env.get("OUTPUT_FOLDER") or "scripts/embedding/mteb")
                output_folder = (REPO_ROOT / output_folder).resolve() if not output_folder.is_absolute() else output_folder
                metrics = _parse_mteb_metrics(output_folder=output_folder, tasks=tasks, model_id=model_id, backend=backend)
            else:
                metrics = {"parse_error": f"no_parser_for_script:{job.script}"}

            merged = dict(base_rec)
            merged["metrics"] = metrics
            merged["tps"] = metrics.get("tps")
            merged["latency_sec"] = metrics.get("latency_sec")
            merged["backend"] = backend
            merged["model"] = model
            merged["model_id"] = model_id

            _write_json(metrics_path, merged)
            _append_jsonl(summary_jsonl, merged)

            csv_rows.append(
                {
                    "run_id": run_id,
                    "job_name": job.name,
                    "script": job.script,
                    "exit_code": exit_code,
                    "backend": backend,
                    "model": model,
                    "model_id": model_id,
                    "tps": merged.get("tps"),
                    "latency_sec": merged.get("latency_sec"),
                    "avg_batch_time_sec": metrics.get("avg_batch_time_sec"),
                    "count": metrics.get("count"),
                    "num_batches": metrics.get("num_batches"),
                    "started_at_utc": started_s,
                    "ended_at_utc": ended_s,
                    "log_path": str(log_path),
                    "metrics_path": str(metrics_path),
                }
            )

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
