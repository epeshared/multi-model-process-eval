#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running directly without installing package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def _norm_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    b = str(base_url).strip()
    if not b:
        return None
    if b.startswith("http://") or b.startswith("https://"):
        return b
    return f"http://{b}"


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _parse_size(size: str) -> Tuple[int, int]:
    s = (size or "").strip().lower()
    if not s:
        return (0, 0)
    if "x" not in s:
        v = int(s)
        return (v, v)
    a, b = s.split("x", 1)
    return (int(a), int(b))


def _collect_images(images_dir: Path, *, image_size: str, max_samples: int) -> List[str]:
    if not images_dir.exists():
        raise SystemExit(f"images_dir not found: {images_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"images_dir is not a directory: {images_dir}")

    # Accept common image extensions.
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not all_files:
        raise SystemExit(f"no images found under: {images_dir}")

    size_tag = (image_size or "").strip().lower()
    if size_tag:
        # Our generator uses file names like: img_1280x720_00012.png
        filtered = [p for p in all_files if size_tag in p.name.lower()]
        if filtered:
            all_files = filtered

    # Deterministic ordering.
    all_files = sorted(all_files)

    if max_samples > 0:
        all_files = all_files[: max_samples]

    return [str(p) for p in all_files]


def parse_args(argv: Any = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark image embeddings (VL embedding)")
    ap.add_argument(
        "--backend",
        default="sglang",
        choices=["sglang", "vllm-http", "vllm"],
        help="Backend (sglang HTTP or vLLM OpenAI-compatible HTTP)",
    )
    ap.add_argument("--model-id", required=True, help="Served model name / model_id for /v1/embeddings")
    ap.add_argument("--base-url", required=True, help="SGLang server base URL, e.g. http://127.0.0.1:30000")
    ap.add_argument("--api", default="v1", choices=["v1", "openai"], help="SGLang embedding API mode")
    ap.add_argument("--api-key", default="", help="API key if needed")
    ap.add_argument("--timeout", type=float, default=900.0, help="HTTP timeout per request")
    ap.add_argument(
        "--image-transport",
        default="data-url",
        choices=["data-url", "base64", "path/url"],
        help="How to send images to the server",
    )

    ap.add_argument("--images-dir", required=True, help="Directory containing local images")
    ap.add_argument("--image-size", default="", help="Optional filter tag like 512x512 (matched in file name)")
    ap.add_argument("--max-samples", type=int, default=1000, help="Max images to embed")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size")
    ap.add_argument("--warmup-samples", type=int, default=1, help="Warmup sample count (<=1 disables)")
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--print-model-info", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)

    from src.tasks.embedding import embed_with_session, load_embedding_session

    base_url = _norm_base_url(args.base_url)
    if not base_url:
        raise SystemExit("--base-url is required")

    images_dir = Path(str(args.images_dir)).expanduser().resolve()
    max_samples = int(args.max_samples)
    if max_samples <= 0:
        raise SystemExit("--max-samples must be > 0")

    images = _collect_images(images_dir, image_size=str(args.image_size or ""), max_samples=max_samples)

    backend = str(args.backend or "").strip().lower()
    if backend in {"vllm", "vllm_openai", "vllm-http-openai", "vllm_openai_http"}:
        backend = "vllm-http"

    session = load_embedding_session(
        model_id=str(args.model_id),
        backend_name=backend,
        base_url=base_url,
        api=str(args.api),
        api_key=str(args.api_key or ""),
        timeout=float(args.timeout),
        image_transport=str(args.image_transport),
        print_model_info=bool(args.print_model_info),
    )

    def _warmup() -> None:
        n = int(args.warmup_samples)
        if n <= 1:
            return
        warm = images[: min(n, len(images))]
        if not warm:
            return
        _ = embed_with_session(
            session,
            inputs=warm,
            modality="image",
            batch_size=min(int(args.batch_size), max(1, len(warm))),
            max_length=0,
            normalize=bool(args.normalize),
            profile=False,
        )

    _warmup()

    t0 = time.time()
    emb = embed_with_session(
        session,
        inputs=images,
        modality="image",
        batch_size=int(args.batch_size),
        max_length=0,
        normalize=bool(args.normalize),
        profile=False,
    )
    t1 = time.time()

    elapsed = float(t1 - t0)
    bs = max(1, int(args.batch_size))
    num_batches = int(math.ceil(len(images) / bs)) if images else 0

    rec: Dict[str, Any] = {
        "dataset": "local_images",
        "modality": "image",
        "images_dir": str(images_dir),
        "image_size": str(args.image_size or ""),
        "count": len(images),
        "time_sec": elapsed,
        "tps": (len(images) / elapsed) if elapsed > 0 else float("inf"),
        "num_batches": num_batches,
        "avg_batch_time_sec": (elapsed / num_batches) if num_batches > 0 else 0.0,
        "batch_size": int(args.batch_size),
        "backend": str(args.backend),
        "model_id": str(args.model_id),
        "image_transport": str(args.image_transport),
        "api": str(args.api),
        "timeout": float(args.timeout),
        "embedding_shape": list(getattr(emb, "shape", ())),
    }

    print(json.dumps(rec, indent=2, default=str))


if __name__ == "__main__":
    main()
