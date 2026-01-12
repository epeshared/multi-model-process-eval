#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running directly without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

QWEN3_MODELS: List[str] = [
    "qwen3-0.6b",
    "qwen3-1.7b",
    "qwen3-4b",
]

MODEL_ID_MAP: Dict[str, str] = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
}

QWEN3_BACKENDS: List[str] = [
    "sglang",
    "vllm-http",
]

DATASETS: List[str] = [
    "single",
    "synthetic",
]


def _gen_synthetic_token_texts(num_samples: int, token_len: int, seed: int) -> List[str]:
    import random

    if num_samples <= 0:
        return []
    if token_len <= 0:
        raise ValueError("--synthetic-token-len must be > 0 when --dataset=synthetic")

    rng = random.Random(int(seed))
    out: List[str] = []
    for _ in range(int(num_samples)):
        toks = [f"w{rng.randint(0, 999999)}" for _ in range(int(token_len))]
        out.append(" ".join(toks))
    return out


def parse_args(argv: Any = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3 (text-only) entry point")
    p.add_argument("--model", required=True, choices=QWEN3_MODELS)
    p.add_argument("--model-id", help="Optional override with HF repo id or local model path")
    p.add_argument("--backend", default="vllm-http", choices=QWEN3_BACKENDS)

    p.add_argument("--dataset", default="synthetic", choices=DATASETS)

    # single
    p.add_argument("--prompt", default="Write a short sentence.")

    # synthetic
    p.add_argument("--synthetic-num-prompts", type=int, default=10)
    p.add_argument("--synthetic-token-len", type=int, default=32)
    p.add_argument("--synthetic-seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=1)

    # runtime
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=0)

    # HTTP backends
    p.add_argument("--base-url", help="Server base URL (for backend=sglang or backend=vllm-http)")
    p.add_argument("--api-key", default="")
    p.add_argument("--timeout", type=float, default=600.0)

    return p.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)

    from src.tasks.qwen3 import chat_with_session, load_qwen3_session

    model_id = args.model_id or MODEL_ID_MAP.get(args.model, args.model)

    session = load_qwen3_session(
        model_id,
        backend_name=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=float(args.timeout),
        # For OpenAI-compatible servers, use the logical model name as served model.
        served_model=(args.model if (args.backend or "").lower() in {"vllm-http"} else ""),
    )

    warmup = max(0, int(getattr(args, "warmup", 0) or 0))

    def _run_batch(prompts: List[str]) -> List[str]:
        return chat_with_session(session, prompt=prompts, max_new_tokens=int(args.max_new_tokens))

    if args.dataset == "single":
        if warmup > 0:
            for _ in range(warmup):
                _ = _run_batch([args.prompt])

        t0 = time.time()
        out = _run_batch([args.prompt])
        t1 = time.time()
        print(
            json.dumps(
                {
                    "dataset": "single",
                    "model_id": model_id,
                    "backend": args.backend,
                    "time_sec": (t1 - t0),
                    "outputs": out,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.dataset == "synthetic":
        n = int(getattr(args, "synthetic_num_prompts", 0) or 0)
        if n <= 0:
            raise ValueError("--synthetic-num-prompts must be > 0")
        batch_size = int(getattr(args, "batch_size", 1) or 1)
        if batch_size <= 0:
            raise ValueError("--batch-size must be > 0")

        prompts = _gen_synthetic_token_texts(
            num_samples=n,
            token_len=int(getattr(args, "synthetic_token_len", 0) or 0),
            seed=int(getattr(args, "synthetic_seed", 1234) or 1234),
        )

        if warmup > 0:
            for _ in range(warmup):
                _ = _run_batch(prompts[:1])

        t0 = time.time()
        outs: List[str] = []
        num_batches = 0
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            num_batches += 1
            outs.extend(_run_batch(batch))
        t1 = time.time()

        total_time = (t1 - t0)
        count = len(prompts)
        time_per_batch = (total_time / num_batches) if num_batches > 0 else float("inf")
        time_per_sample = (total_time / count) if count > 0 else float("inf")

        print(
            json.dumps(
                {
                    "dataset": "synthetic",
                    "count": count,
                    "token_len": int(getattr(args, "synthetic_token_len", 0) or 0),
                    "batch_size": batch_size,
                    "model_id": model_id,
                    "backend": args.backend,
                    "max_new_tokens": int(args.max_new_tokens),
                    "time_sec": total_time,
                    "time_per_batch_sec": time_per_batch,
                    "time_per_sample_sec": time_per_sample,
                    "outputs_preview": outs[: min(3, len(outs))],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
