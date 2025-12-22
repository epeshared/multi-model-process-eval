#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import signal
import time
from typing import Any, Dict, List, Optional


# Allow running this script directly without installing the package.
# When executed as `python scripts/py/run_vl.py`, Python adds `scripts/py` to sys.path,
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
]


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

    # generation/runtime
    p.add_argument("--max-new-tokens", type=int, default=128)
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

    # output
    p.add_argument("--output-jsonl", help="Optional path to save per-sample outputs (jsonl)")
    return p.parse_args(argv)


def _norm_paths(args: argparse.Namespace) -> None:
    # Back-compat: allow --dataset-path to stand in for captions file
    if args.dataset == "flickr8k":
        if not args.flickr8k_captions_file and args.dataset_path:
            args.flickr8k_captions_file = args.dataset_path
        if not args.flickr8k_images_dir and args.dataset_path:
            # common pattern: captions file under dataset root
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

    if args.dataset == "single":
        if not args.image:
            raise ValueError("--image is required when --dataset=single")
        out = chat_with_session(
            session,
            image_paths=args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
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
        # Timing excludes model/client load; session was created above.
        t0 = time.time()

        rows: List[Dict[str, Any]] = []
        for i in range(0, n, bs):
            batch_paths = ds.image_paths[i : i + bs]
            outputs = chat_with_session(
                session,
                image_paths=batch_paths,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
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
            "seconds_per_batch": (elapsed / (n/bs)) if n > 0 else 0.0,
            "model_id": model_id,
            "backend": args.backend,
        }

        if args.output_jsonl:
            os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
            with open(args.output_jsonl, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rec["output_jsonl"] = args.output_jsonl

        print(json.dumps(rec, indent=2, ensure_ascii=False))
        return

    raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
