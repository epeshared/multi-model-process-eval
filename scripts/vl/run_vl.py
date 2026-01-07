#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import signal
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running this script directly without installing the package.
# When executed as `python scripts/vl/run_vl.py`, Python adds `scripts/vl` to sys.path,
# but not the repository root, so `import src...` would fail.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

VL_MODELS: List[str] = [
    "qwen2.5-vl-7b-instruct",
]

MODEL_ID_MAP: Dict[str, str] = {
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
}

VL_BACKENDS: List[str] = [
    "torch",
    "sglang",
    "sglang-offline",
    "vllm",
    "vllm-http",
]

DATASETS: List[str] = [
    "single",
    "flickr8k",
    "synthetic",  # NEW
]


def _csv_to_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    parts = [x.strip() for x in str(s).split(",")]
    parts = [x for x in parts if x]
    return parts or None


def _parse_hw(s: Optional[str], default_hw: str = "224x224") -> Tuple[int, int]:
    """Parse image size string into (H, W)."""
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
    """Generate N random RGB images of size HxW, saved as PNG files."""
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


# NOTE: rest of this file is identical to the previously-used runner (moved),
# kept in place to preserve behavior for existing shell scripts.


def parse_args(argv: Any = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision-language (VL) entry point")
    p.add_argument("--model", required=True, choices=VL_MODELS, help="Logical model key")
    p.add_argument("--model-id", help="Optional override with HF repo id or local model path")
    p.add_argument("--backend", default="torch", choices=VL_BACKENDS)

    p.add_argument("--dataset", default="single", choices=DATASETS)

    # single
    p.add_argument("--image", help="Path to image (dataset=single)")
    p.add_argument("--prompt", default="Describe the image.")

    # flickr8k
    p.add_argument("--dataset-path", help="Path to Flickr8k.token.txt (dataset=flickr8k)")
    p.add_argument("--flickr8k-images-dir", help="Flickr8k images directory")
    p.add_argument("--flickr8k-captions-file", help="Flickr8k.token.txt path")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)

    # synthetic
    p.add_argument("--synthetic-image-size", default="224x224", help="e.g. 224x224 or 224,224 or '224 224'")
    p.add_argument("--synthetic-num-images", type=int, default=50, help="How many random images to generate")
    p.add_argument("--synthetic-seed", type=int, default=1234, help="RNG seed for synthetic images")
    p.add_argument(
        "--synthetic-out-dir",
        default="",
        help="Where to save synthetic images; empty => use a temp dir",
    )

    # generation/runtime
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup calls to run after loading the session (excluded from timing)",
    )
    p.add_argument("--device", help="Device id, e.g., cuda:0")
    p.add_argument("--dtype", help="auto|fp16|bf16|fp32")
    p.add_argument(
        "--use-amx",
        action="store_true",
        default=False,
        help="Enable AMX/IPEX acceleration for torch CPU runs (requires intel_extension_for_pytorch)",
    )
    p.add_argument(
        "--print-model-info",
        action="store_true",
        default=False,
        help="Print backend/model info during session load",
    )

    # HTTP backends
    p.add_argument("--base-url", help="Server base URL (for backend=sglang or backend=vllm-http)")
    p.add_argument("--api", default="v1", help="API mode (reserved; v1 recommended)")
    p.add_argument("--api-key", default="", help="API key for HTTP backends")
    p.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout seconds")
    p.add_argument(
        "--image-transport",
        default="data-url",
        help="Image transport for HTTP backends: data-url|path/url",
    )

    # Offline backends
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for offline backends")
    p.add_argument("--dp-size", type=int, default=1, help="Data parallel size for sglang-offline")
    p.add_argument("--max-model-len", type=int, default=8192, help="Max model length for vLLM offline")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory util for vLLM offline")
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument("--attn-implementation", help="eager|sdpa|flash_attention_2")

    # PROFILING
    p.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable profiling for sglang backends (HTTP: /start_profile;/stop_profile, offline: torch.profiler)",
    )
    p.add_argument(
        "--profile-record-shapes",
        action="store_true",
        default=True,
        help="Profiler: record operator shapes (may increase overhead)",
    )
    p.add_argument(
        "--profile-activities",
        default="CPU,CUDA",
        help="Profiler activities CSV, e.g. CPU,CUDA",
    )
    p.add_argument(
        "--profile-out-dir",
        default="",
        help="(offline torch.profiler) export chrome trace json to this dir; empty disables export",
    )
    p.add_argument(
        "--profile-out-name",
        default="vl_profile",
        help="(offline torch.profiler) chrome trace base name",
    )
    p.add_argument(
        "--profile-strict",
        action="store_true",
        default=False,
        help="If set, fail fast when profiler cannot start",
    )

    # output
    p.add_argument("--output-jsonl", help="Optional path to save per-sample outputs (jsonl)")
    return p.parse_args(argv)


def _norm_paths(args: argparse.Namespace) -> None:
    if args.dataset == "flickr8k":
        if not args.flickr8k_captions_file and args.dataset_path:
            args.flickr8k_captions_file = args.dataset_path
        if not args.flickr8k_images_dir and args.dataset_path:
            args.flickr8k_images_dir = os.path.dirname(args.dataset_path)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    _norm_paths(args)

    from src.tasks.vl import chat_with_session, load_vl_session

    model_id = args.model_id or MODEL_ID_MAP.get(args.model, args.model)

    backend = (args.backend or "torch").lower()
    backend_kwargs: Dict[str, Any] = {}
    if backend in {"vllm", "vllm-offline"}:
        backend_kwargs.update(
            {
                "tensor_parallel_size": int(args.tp_size),
                "max_model_len": int(args.max_model_len),
                "gpu_memory_utilization": float(args.gpu_memory_utilization),
            }
        )
    elif backend in {"sglang-offline", "sglang_offline"}:
        backend_kwargs.update({"tp_size": int(args.tp_size), "dp_size": int(args.dp_size)})

    profile_kwargs: Dict[str, Any] = {}
    if bool(args.profile):
        acts = _csv_to_list(getattr(args, "profile_activities", "CPU,CUDA"))
        if acts:
            profile_kwargs["activities"] = acts
        profile_kwargs["record_shapes"] = bool(getattr(args, "profile_record_shapes", True))
        out_dir = str(getattr(args, "profile_out_dir", "") or "").strip()
        if out_dir:
            profile_kwargs["out_dir"] = out_dir
        profile_kwargs["out_name"] = str(getattr(args, "profile_out_name", "vl_profile") or "vl_profile")
        profile_kwargs["strict"] = bool(getattr(args, "profile_strict", False))

    session = load_vl_session(
        model_id,
        backend_name=args.backend,
        device=args.device,
        dtype=args.dtype,
        use_amx=bool(args.use_amx),
        print_model_info=bool(args.print_model_info),
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        base_url=args.base_url,
        api=args.api,
        api_key=args.api_key,
        timeout=float(args.timeout),
        image_transport=args.image_transport,
        **backend_kwargs,
    )

    warmup = max(0, int(getattr(args, "warmup", 0) or 0))

    if args.dataset == "single":
        if not args.image:
            raise ValueError("--image is required when --dataset=single")

        if warmup > 0:
            print(f"[vl.warmup] start dataset=single warmup={warmup} image={args.image}", flush=True)
            for _ in range(warmup):
                _ = chat_with_session(
                    session,
                    image_paths=args.image,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    profile=False,
                    profile_kwargs=None,
                )
            print("[vl.warmup] done", flush=True)

        print("[vl.run] start dataset=single", flush=True)
        out = chat_with_session(
            session,
            image_paths=args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            profile=bool(args.profile),
            profile_kwargs=(profile_kwargs if args.profile else None),
        )
        print(json.dumps({"outputs": out}, indent=2, ensure_ascii=False))
        return

    if args.dataset == "flickr8k":
        from src.data import load_flickr8k

        captions_file = args.flickr8k_captions_file
        images_dir = args.flickr8k_images_dir or ""
        if not captions_file:
            raise ValueError("--flickr8k-captions-file (or --dataset-path) is required for dataset=flickr8k")

        ds = load_flickr8k(
            images_dir=images_dir,
            captions_file=captions_file,
            captions_per_image=1,
            modality="image",
            max_images=args.max_samples,
        )

        bs = max(1, int(args.batch_size))
        n = len(ds.image_paths)

        if warmup > 0 and n > 0:
            warm_paths = ds.image_paths[: min(bs, n)]
            print(
                f"[vl.warmup] start dataset=flickr8k warmup={warmup} warmup_batch={len(warm_paths)} batch_size={bs}",
                flush=True,
            )
            for _ in range(warmup):
                _ = chat_with_session(
                    session,
                    image_paths=warm_paths,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    profile=False,
                    profile_kwargs=None,
                )
            print("[vl.warmup] done", flush=True)

        print(
            f"[vl.run] start timing dataset=flickr8k count={n} batch_size={bs} profile={bool(args.profile)}",
            flush=True,
        )
        t0 = time.time()

        rows: List[Dict[str, Any]] = []
        for i in range(0, n, bs):
            batch_paths = ds.image_paths[i : i + bs]
            outputs = chat_with_session(
                session,
                image_paths=batch_paths,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                profile=bool(args.profile),
                profile_kwargs=(profile_kwargs if args.profile else None),
            )
            for pth, txt in zip(batch_paths, outputs):
                rows.append({"image": pth, "prompt": args.prompt, "output": txt})

        t1 = time.time()
        elapsed = t1 - t0
        rec = {
            "dataset": "flickr8k",
            "count": n,
            "batch_size": bs,
            "time_sec": elapsed,
            "samples_per_sec": (n / elapsed) if elapsed > 0 else float("inf"),
            "seconds_per_batch": (elapsed / (n / bs)) if n > 0 else 0.0,
            "model_id": model_id,
            "backend": args.backend,
            "profile": bool(args.profile),
            "profile_kwargs": profile_kwargs if args.profile else None,
        }

        if args.output_jsonl:
            os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
            with open(args.output_jsonl, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rec["output_jsonl"] = args.output_jsonl

        print(json.dumps(rec, indent=2, ensure_ascii=False))
        return

    if args.dataset == "synthetic":
        bs = max(1, int(args.batch_size))
        n = max(0, int(getattr(args, "synthetic_num_images", 0) or 0))
        if n <= 0:
            raise ValueError("--synthetic-num-images must be > 0 for dataset=synthetic")

        h, w = _parse_hw(getattr(args, "synthetic_image_size", "224x224"))

        out_dir = str(getattr(args, "synthetic_out_dir", "") or "").strip()
        tmp_ctx: Optional[tempfile.TemporaryDirectory] = None
        if not out_dir:
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"vl_synth_{h}x{w}_")
            out_dir = tmp_ctx.name

        image_paths = _gen_synthetic_images(
            out_dir=out_dir,
            n=n,
            h=h,
            w=w,
            seed=int(getattr(args, "synthetic_seed", 1234)),
        )

        if warmup > 0 and len(image_paths) > 0:
            warm_paths = image_paths[: min(bs, len(image_paths))]
            print(
                f"[vl.warmup] start dataset=synthetic warmup={warmup} warmup_batch={len(warm_paths)} "
                f"batch_size={bs} image_size={h}x{w}",
                flush=True,
            )
            for _ in range(warmup):
                _ = chat_with_session(
                    session,
                    image_paths=warm_paths,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    profile=False,
                    profile_kwargs=None,
                )
            print("[vl.warmup] done", flush=True)

        print(
            f"[vl.run] start timing dataset=synthetic count={len(image_paths)} batch_size={bs} "
            f"image_size={h}x{w} profile={bool(args.profile)}",
            flush=True,
        )
        t0 = time.time()

        rows: List[Dict[str, Any]] = []
        for i in range(0, len(image_paths), bs):
            batch_paths = image_paths[i : i + bs]
            outputs = chat_with_session(
                session,
                image_paths=batch_paths,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                profile=bool(args.profile),
                profile_kwargs=(profile_kwargs if args.profile else None),
            )
            for pth, txt in zip(batch_paths, outputs):
                rows.append(
                    {
                        "image": pth,
                        "prompt": args.prompt,
                        "output": txt,
                        "synthetic_image_size": f"{h}x{w}",
                    }
                )

        t1 = time.time()
        elapsed = t1 - t0
        rec = {
            "dataset": "synthetic",
            "count": len(image_paths),
            "batch_size": bs,
            "image_size": f"{h}x{w}",
            "time_sec": elapsed,
            "samples_per_sec": (len(image_paths) / elapsed) if elapsed > 0 else float("inf"),
            "seconds_per_batch": (elapsed / (len(image_paths) / bs)) if len(image_paths) > 0 else 0.0,
            "model_id": model_id,
            "backend": args.backend,
            "profile": bool(args.profile),
            "profile_kwargs": profile_kwargs if args.profile else None,
            "synthetic_out_dir": out_dir,
            "synthetic_seed": int(getattr(args, "synthetic_seed", 1234)),
        }

        if args.output_jsonl:
            os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
            with open(args.output_jsonl, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rec["output_jsonl"] = args.output_jsonl

        print(json.dumps(rec, indent=2, ensure_ascii=False))

        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return

    raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
