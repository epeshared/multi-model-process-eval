#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests  # type: ignore

    _REQUESTS_OK = True
except Exception:
    _REQUESTS_OK = False


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    messages: List[Dict[str, Any]]


def _norm_base_url(base_url: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        raise ValueError("--base-url is required")
    return b


def _chat_url(base_url: str, endpoint: str) -> str:
    b = _norm_base_url(base_url)
    ep = (endpoint or "").strip()
    if not ep:
        ep = "/v1/chat/completions"
    if not ep.startswith("/"):
        ep = "/" + ep
    # If base_url already includes /v1 and endpoint also starts with /v1, avoid /v1/v1.
    if b.endswith("/v1") and ep.startswith("/v1/"):
        return b + ep[len("/v1") :]
    return b + ep


def _quantile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    if q <= 0:
        return float(min(xs))
    if q >= 1:
        return float(max(xs))
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    if n == 1:
        return ys[0]

    # Linear interpolation between closest ranks (like numpy default).
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def _mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _safe_stdev(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return float(statistics.stdev(xs))


def _gen_random_prompt_words(num_words: int, rng: random.Random) -> str:
    # Words are a rough proxy of tokens; keeps prompt generation lightweight.
    # Using a fixed prefix helps avoid empty/too-short prompts.
    words = ["w" + str(rng.randint(0, 999999)) for _ in range(max(1, int(num_words)))]
    return " ".join(words)


def _load_requests_from_file(path: Path) -> List[RequestSpec]:
    # Supported:
    # - .txt: each line is a prompt -> messages=[{"role":"user","content": line}]
    # - .jsonl: each line is JSON; supports keys:
    #     - {"prompt": "..."}
    #     - {"text": "..."}
    #     - {"messages": [...]} (OpenAI chat format)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    reqs: List[RequestSpec] = []
    if p.suffix.lower() == ".txt":
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                s = line.strip("\n")
                if not s.strip():
                    continue
                reqs.append(RequestSpec(request_id=len(reqs), messages=[{"role": "user", "content": s}]))
        return reqs

    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    continue
                if isinstance(obj.get("messages"), list):
                    msgs = obj["messages"]
                    reqs.append(RequestSpec(request_id=len(reqs), messages=[dict(m) for m in msgs]))
                    continue
                prompt = obj.get("prompt", obj.get("text"))
                if isinstance(prompt, str) and prompt.strip():
                    reqs.append(RequestSpec(request_id=len(reqs), messages=[{"role": "user", "content": prompt}]))
        return reqs

    raise ValueError(f"Unsupported prompts file type: {p.suffix} (use .txt or .jsonl)")


def _headers(api_key: str) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _post_chat_stream(
    *,
    url: str,
    payload: Dict[str, Any],
    api_key: str,
    timeout: float,
) -> Tuple[str, Dict[str, Any]]:
    """Return (text, metrics).

    Metrics:
      - ttft_sec: seconds until first non-empty delta.content
      - total_sec: wall time
      - usage: dict (best-effort)
    """

    if not _REQUESTS_OK:
        raise RuntimeError("requests is required: pip install requests")

    started = time.time()
    ttft_sec: Optional[float] = None
    text_parts: List[str] = []
    usage: Dict[str, Any] = {}

    with requests.post(
        url,
        headers=_headers(api_key),
        json=payload,
        timeout=timeout,
        stream=True,
    ) as resp:
        if resp.status_code >= 400:
            err_msg = resp.text
            try:
                j = resp.json()
                if isinstance(j, dict):
                    e = j.get("error")
                    if isinstance(e, dict) and e.get("message"):
                        err_msg = str(e.get("message"))
            except Exception:
                pass
            raise RuntimeError(f"HTTP {resp.status_code}: {err_msg}")

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break

            try:
                evt = json.loads(data_str)
            except Exception:
                continue

            if not isinstance(evt, dict):
                continue

            # vLLM can optionally include usage in-stream.
            u = evt.get("usage")
            if isinstance(u, dict):
                usage = u

            choices = evt.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = (choices[0] or {}).get("delta")
            if not isinstance(delta, dict):
                continue
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                if ttft_sec is None:
                    ttft_sec = time.time() - started
                text_parts.append(piece)

    total_sec = time.time() - started
    return "".join(text_parts), {"ttft_sec": ttft_sec, "total_sec": total_sec, "usage": usage}


def _post_chat_non_stream(
    *,
    url: str,
    payload: Dict[str, Any],
    api_key: str,
    timeout: float,
) -> Tuple[str, Dict[str, Any]]:
    if not _REQUESTS_OK:
        raise RuntimeError("requests is required: pip install requests")

    started = time.time()
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
    if resp.status_code >= 400:
        err_msg = resp.text
        try:
            j = resp.json()
            if isinstance(j, dict):
                e = j.get("error")
                if isinstance(e, dict) and e.get("message"):
                    err_msg = str(e.get("message"))
        except Exception:
            pass
        raise RuntimeError(f"HTTP {resp.status_code}: {err_msg}")

    data = resp.json()
    total_sec = time.time() - started
    usage = data.get("usage") if isinstance(data, dict) else {}
    out_text = ""
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = (choices[0] or {}).get("message")
            if isinstance(msg, dict):
                c = msg.get("content")
                out_text = c if isinstance(c, str) else str(c)

    return out_text, {"ttft_sec": None, "total_sec": total_sec, "usage": usage if isinstance(usage, dict) else {}}


def _tpot_from_metrics(m: Dict[str, Any]) -> Optional[float]:
    """TPOT = (total - ttft) / max(1, completion_tokens - 1)."""
    try:
        ttft = m.get("ttft_sec")
        total = m.get("total_sec")
        usage = m.get("usage") or {}
        comp = usage.get("completion_tokens")
        if ttft is None or total is None or comp is None:
            return None
        comp = int(comp)
        denom = max(1, comp - 1)
        return (float(total) - float(ttft)) / float(denom)
    except Exception:
        return None


def parse_args(argv: Any = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark a vLLM OpenAI-compatible chat server (TTFT/TPOT/QPS/P50/P99)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM server base URL")
    p.add_argument("--endpoint", default="/v1/chat/completions", help="Chat completions endpoint")
    p.add_argument("--model", default="qwen3-0.6b", help="Served model name (must match --served-model-name)")
    p.add_argument("--api-key", default="", help="Optional bearer token")
    p.add_argument("--timeout", type=float, default=600.0)

    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument(
        "--request-rate",
        type=float,
        default=0.0,
        help="If >0, pace request starts to this global rate (req/sec). 0 means fire as fast as possible.",
    )

    # Prompt source
    p.add_argument("--prompts-file", default="", help=".txt or .jsonl prompts file")
    p.add_argument("--random-input-len", type=int, default=256, help="Approx words per prompt when not using --prompts-file")
    p.add_argument("--seed", type=int, default=1234)

    # Generation
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Use streaming (required for TTFT).",
    )
    p.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming (TTFT=null)")

    p.add_argument("--save-json", default="", help="Optional path to save raw per-request results + summary")
    p.add_argument("--quiet", action="store_true", help="Less stdout")

    return p.parse_args(argv)


def _build_specs(args: argparse.Namespace) -> List[RequestSpec]:
    if args.prompts_file:
        reqs = _load_requests_from_file(Path(args.prompts_file))
        if not reqs:
            raise ValueError("--prompts-file produced 0 requests")
        if len(reqs) >= int(args.num_prompts):
            return [RequestSpec(request_id=i, messages=reqs[i].messages) for i in range(int(args.num_prompts))]
        # Repeat if file is shorter than num-prompts
        out: List[RequestSpec] = []
        i = 0
        while len(out) < int(args.num_prompts):
            out.append(RequestSpec(request_id=len(out), messages=reqs[i % len(reqs)].messages))
            i += 1
        return out

    rng = random.Random(int(args.seed))
    out = []
    for i in range(int(args.num_prompts)):
        prompt = _gen_random_prompt_words(int(args.random_input_len), rng)
        out.append(RequestSpec(request_id=i, messages=[{"role": "user", "content": prompt}]))
    return out


def _worker(
    *,
    spec: RequestSpec,
    scheduled_start: float,
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
    api_key: str,
    timeout: float,
) -> Dict[str, Any]:
    # Rate limiting / scheduling
    now = time.time()
    if scheduled_start > now:
        time.sleep(scheduled_start - now)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": spec.messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    try:
        if stream:
            text, m = _post_chat_stream(url=url, payload=payload, api_key=api_key, timeout=timeout)
        else:
            text, m = _post_chat_non_stream(url=url, payload=payload, api_key=api_key, timeout=timeout)

        tpot = _tpot_from_metrics(m)
        usage = m.get("usage") if isinstance(m.get("usage"), dict) else {}
        return {
            "ok": True,
            "request_id": spec.request_id,
            "ttft_sec": m.get("ttft_sec"),
            "total_sec": m.get("total_sec"),
            "tpot_sec_per_token": tpot,
            "usage": usage,
            "output_len_chars": len(text),
        }
    except Exception as e:
        return {
            "ok": False,
            "request_id": spec.request_id,
            "error": str(e),
        }


def main(argv: Any = None) -> None:
    args = parse_args(argv)

    if not _REQUESTS_OK:
        print("ERROR: requests is not installed. Install it with: pip install requests", file=sys.stderr)
        sys.exit(2)

    num_prompts = int(args.num_prompts)
    concurrency = max(1, int(args.concurrency))
    if num_prompts <= 0:
        raise ValueError("--num-prompts must be > 0")
    if concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    url = _chat_url(args.base_url, args.endpoint)
    specs = _build_specs(args)

    # Build scheduled start times
    t0 = time.time()
    if float(args.request_rate) and float(args.request_rate) > 0:
        rate = float(args.request_rate)
        schedule = [t0 + (i / rate) for i in range(num_prompts)]
    else:
        schedule = [t0 for _ in range(num_prompts)]

    if not args.quiet:
        print(f"[bench] url={url}")
        print(f"[bench] model={args.model}")
        print(f"[bench] num_prompts={num_prompts} concurrency={concurrency} request_rate={float(args.request_rate) or 0}")
        print(f"[bench] stream={bool(args.stream)} max_tokens={int(args.max_tokens)} temperature={float(args.temperature)}")

    # Run
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Dict[str, Any]] = []

    started_wall = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = []
        for i in range(num_prompts):
            futs.append(
                ex.submit(
                    _worker,
                    spec=specs[i],
                    scheduled_start=schedule[i],
                    url=url,
                    model=str(args.model),
                    max_tokens=int(args.max_tokens),
                    temperature=float(args.temperature),
                    stream=bool(args.stream),
                    api_key=str(args.api_key or ""),
                    timeout=float(args.timeout),
                )
            )

        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            if not args.quiet and (done % max(1, (num_prompts // 10))) == 0:
                ok = sum(1 for x in results if x.get("ok"))
                print(f"[bench] progress {done}/{num_prompts} ok={ok}")

    ended_wall = time.time()

    oks = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]

    total_secs = [float(r["total_sec"]) for r in oks if r.get("total_sec") is not None]
    ttft_secs = [float(r["ttft_sec"]) for r in oks if r.get("ttft_sec") is not None]
    tpot_secs = [float(r["tpot_sec_per_token"]) for r in oks if r.get("tpot_sec_per_token") is not None]

    duration_sec = max(1e-9, ended_wall - started_wall)
    qps = (len(oks) / duration_sec) if duration_sec > 0 else None

    summary: Dict[str, Any] = {
        "ok": True,
        "url": url,
        "model": str(args.model),
        "num_prompts": num_prompts,
        "concurrency": concurrency,
        "request_rate": float(args.request_rate) if float(args.request_rate) > 0 else 0.0,
        "stream": bool(args.stream),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "duration_sec": duration_sec,
        "success": len(oks),
        "errors": len(errs),
        "qps": qps,
        "latency_sec": {
            "avg": _mean(total_secs),
            "stdev": _safe_stdev(total_secs),
            "p50": _quantile(total_secs, 0.50),
            "p99": _quantile(total_secs, 0.99),
        },
        "ttft_sec": {
            "avg": _mean(ttft_secs),
            "stdev": _safe_stdev(ttft_secs),
            "p50": _quantile(ttft_secs, 0.50),
            "p99": _quantile(ttft_secs, 0.99),
        },
        "tpot_sec_per_token": {
            "avg": _mean(tpot_secs),
            "stdev": _safe_stdev(tpot_secs),
            "p50": _quantile(tpot_secs, 0.50),
            "p99": _quantile(tpot_secs, 0.99),
        },
    }

    out_obj = {"summary": summary, "results": results}

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Print a compact summary line for dashboards.
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Non-zero exit if nothing succeeded.
    if len(oks) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
