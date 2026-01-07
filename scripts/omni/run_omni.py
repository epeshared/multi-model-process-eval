#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running this script directly without installing the package.
# When executed as `python scripts/omni/run_omni.py`, Python adds `scripts/omni` to sys.path,
# but not the repository root, so `import src...` would fail.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

OMNI_MODELS: List[str] = [
    "qwen2.5-omni-7b",
]

MODEL_ID_MAP: Dict[str, str] = {
    "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",
}

OMNI_BACKENDS: List[str] = [
    "sglang",
    "vllm",
    "vllm-http",
]

DATASETS: List[str] = [
    "single",
    "synthetic",
]


def _csv_to_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    parts = [x.strip() for x in str(s).split(",")]
    parts = [x for x in parts if x]
    return parts or None


def _parse_hw(s: Optional[str], default_hw: str = "224x224") -> Tuple[int, int]:
    raw = (s or "").strip() or default_hw
    raw = raw.lower().replace("*", "x")

    for sep in ["x", ",", " "]:
        if sep in raw:
            parts = [p for p in raw.split(sep) if p.strip()]
            if len(parts) == 2:
                h = int(parts[0].strip())
                w = int(parts[1].strip())
                if h <= 0 or w <= 0:
                    raise ValueError(f"Invalid synthetic image size: {s}")
                return (h, w)

    if raw.isdigit():
        v = int(raw)
        if v <= 0:
            raise ValueError(f"Invalid synthetic image size: {s}")
        return (v, v)

    raise ValueError(f"Invalid synthetic image size format: {s} (expected 224x224 / 224,224 / '224 224')")


def _gen_synthetic_images(out_dir: str, n: int, h: int, w: int, seed: int) -> List[str]:
    import numpy as np
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    paths: List[str] = []
    for i in range(int(n)):
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        p = os.path.join(out_dir, f"synthetic_{h}x{w}_{i:06d}.png")
        img.save(p)
        paths.append(p)
    return paths


def parse_args(argv: Any = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Omni (multimodal) entry point")
    p.add_argument("--model", required=True, choices=OMNI_MODELS)
    p.add_argument("--model-id", help="Optional override with HF repo id or local model path")
    p.add_argument("--backend", default="sglang", choices=OMNI_BACKENDS)

    p.add_argument("--dataset", default="single", choices=DATASETS)

    # single
    p.add_argument("--image", help="Path to image (dataset=single)")
    p.add_argument("--prompt", default="Describe the image.")

    # synthetic
    p.add_argument("--synthetic-image-size", default="224x224")
    p.add_argument("--synthetic-num-images", type=int, default=10)
    p.add_argument("--synthetic-seed", type=int, default=1234)
    p.add_argument("--synthetic-out-dir", default="")

    # runtime
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=0)

    # HTTP backends
    p.add_argument("--base-url", help="Server base URL (for backend=sglang or backend=vllm-http)")
    p.add_argument("--api-key", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--image-transport", default="data-url")

    # Offline vLLM
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", help="auto|fp16|bf16|fp32")
    p.add_argument("--device", help="cuda:0|cpu")

    # profiling (sglang-http only)
    p.add_argument("--profile", action="store_true", default=False)
    p.add_argument("--profile-activities", default="CPU,CUDA")
    p.add_argument("--profile-record-shapes", action="store_true", default=True)
    p.add_argument("--profile-out-dir", default="")
    p.add_argument("--profile-out-name", default="omni_profile")
    p.add_argument("--profile-strict", action="store_true", default=False)

    return p.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)

    from src.tasks.omni import chat_with_session, load_omni_session

    model_id = args.model_id or MODEL_ID_MAP.get(args.model, args.model)

    backend = (args.backend or "sglang").lower()
    backend_kwargs: Dict[str, Any] = {}
    if backend in {"vllm", "vllm-offline"}:
        backend_kwargs.update(
            {
                "tensor_parallel_size": int(args.tp_size),
                "max_model_len": int(args.max_model_len),
                "gpu_memory_utilization": float(args.gpu_memory_utilization),
            }
        )

    profile_kwargs: Dict[str, Any] = {}
    if bool(args.profile):
        acts = _csv_to_list(getattr(args, "profile_activities", "CPU,CUDA"))
        if acts:
            profile_kwargs["activities"] = acts
        profile_kwargs["record_shapes"] = bool(getattr(args, "profile_record_shapes", True))
        profile_kwargs["out_dir"] = str(getattr(args, "profile_out_dir", "") or "")
        profile_kwargs["out_name"] = str(getattr(args, "profile_out_name", "omni_profile") or "omni_profile")
        profile_kwargs["strict"] = bool(getattr(args, "profile_strict", False))

    session = load_omni_session(
        model_id,
        backend_name=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=float(args.timeout),
        image_transport=args.image_transport,
        device=args.device,
        dtype=args.dtype,
        **backend_kwargs,
    )

    warmup = max(0, int(getattr(args, "warmup", 0) or 0))

    def _run_once(image_paths: List[str]) -> List[str]:
        return chat_with_session(
            session,
            image_paths=image_paths,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            profile=bool(args.profile),
            profile_kwargs=(profile_kwargs if args.profile else None),
        )

    if args.dataset == "single":
        if not args.image:
            raise ValueError("--image is required when --dataset=single")

        if warmup > 0:
            for _ in range(warmup):
                _ = chat_with_session(session, image_paths=[args.image], prompt=args.prompt, max_new_tokens=8, profile=False)

        t0 = time.time()
        out = _run_once([args.image])
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
        n = int(getattr(args, "synthetic_num_images", 0) or 0)
        if n <= 0:
            raise ValueError("--synthetic-num-images must be > 0")
        h, w = _parse_hw(getattr(args, "synthetic_image_size", "224x224"))

        out_dir = str(getattr(args, "synthetic_out_dir", "") or "").strip()
        tmp_ctx: Optional[tempfile.TemporaryDirectory] = None
        if not out_dir:
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"omni_synth_{h}x{w}_")
            out_dir = tmp_ctx.name

        image_paths = _gen_synthetic_images(out_dir=out_dir, n=n, h=h, w=w, seed=int(args.synthetic_seed))

        if warmup > 0:
            for _ in range(warmup):
                _ = chat_with_session(session, image_paths=image_paths[:1], prompt=args.prompt, max_new_tokens=8, profile=False)

        t0 = time.time()
        outs: List[str] = []
        for pth in image_paths:
            outs.extend(_run_once([pth]))
        t1 = time.time()

        print(
            json.dumps(
                {
                    "dataset": "synthetic",
                    "count": len(image_paths),
                    "image_size": f"{h}x{w}",
                    "time_sec": (t1 - t0),
                    "samples_per_sec": (len(image_paths) / (t1 - t0)) if (t1 - t0) > 0 else float("inf"),
                    "model_id": model_id,
                    "backend": args.backend,
                    "synthetic_out_dir": out_dir,
                    "outputs_preview": outs[: min(len(outs), 3)],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return

    raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
