#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import signal
import os
import sys
import time
import math
from typing import Any, Callable, Dict, List, Tuple

# If output is piped to a command like `head`, stdout may be closed early.
# Default Python behavior can emit noisy BrokenPipeError messages on exit.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

EMBED_MODELS: List[str] = [
    "qwen3-embedding-4b",
    "qwen3-embedding-0.6b",
    "clip-vit-base-patch32",
]

MODEL_ID_MAP: Dict[str, str] = {
    "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
}

EMBED_BACKENDS: List[str] = [
    "torch",
    "sglang",
    "sglang-offline",
    "vllm",
    "vllm-http",
]

DATASET_LOADERS: Dict[str, Dict[str, Any]] = {
    "yahoo_answers": {
        "modality": "text",
    },
    "flickr8k": {
        "modality": "both",
    },
}


def _get_dataset_loader(dataset: str) -> Tuple[Callable[..., List[Any]], str]:
    if dataset == "yahoo_answers":
        from src.data import load_yahoo_answers_jsonl

        return load_yahoo_answers_jsonl, "text"
    raise ValueError(f"Unknown dataset: {dataset}")


def _norm_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    if base_url.startswith("http://") or base_url.startswith("https://"):
        return base_url
    return f"http://{base_url}"


def _shape_list(x: Any) -> List[int] | None:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return list(x.shape)
    except Exception:
        pass
    return None


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding entry point")
    parser.add_argument("--model", required=True, choices=EMBED_MODELS, help="Logical model key")
    parser.add_argument(
        "--model-id",
        help="Optional override with a full HF repo id or local model path; defaults to the preset mapping for --model",
    )
    parser.add_argument("--backend", choices=EMBED_BACKENDS, default="torch")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()), help="Dataset key under src/data")
    parser.add_argument("--dataset-path", required=True, help="Path to the dataset source file (e.g., JSONL)")
    parser.add_argument("--yahoo-mode", choices=["q", "a", "q+a"], default="q", help="Yahoo answers embedding mode")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum samples to load (-1 for all)")

    # Flickr8k dataset options (used when --dataset=flickr8k)
    parser.add_argument("--flickr8k-images-dir", help="Flickr8k images directory (contains *.jpg)")
    parser.add_argument("--flickr8k-captions-file", help="Flickr8k.token.txt path")
    parser.add_argument(
        "--flickr8k-captions-per-image",
        type=int,
        default=1,
        help="How many captions per image to embed (default 1; Flickr8k typically has 5)",
    )
    parser.add_argument(
        "--flickr8k-modality",
        choices=["both", "text", "image"],
        default="both",
        help="Flickr8k embedding modality: both|text|image (default both)",
    )
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    parser.add_argument("--base-url", help="SGLang server base URL (for backend=sglang)")
    parser.add_argument("--api", default="v1", help="SGLang API mode: native|v1|openai")
    parser.add_argument("--api-key", default="", help="API key for SGLang HTTP embedding")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout for SGLang backend")
    parser.add_argument(
        "--image-transport",
        default="data-url",
        help="Image transport for SGLang HTTP embedding: data-url|base64|path/url",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length for torch backend")
    parser.add_argument("--normalize", action="store_true", default=True, help="Whether to L2 normalize outputs")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument(
        "--use-amx",
        action="store_true",
        default=False,
        help="Enable AMX/IPEX acceleration for torch CPU embeddings (requires intel_extension_for_pytorch)",
    )
    parser.add_argument("--encoding-format", help="Encoding format for vLLM HTTP (e.g., bf16)")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for vLLM offline")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max model length for vLLM offline")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilization for vLLM offline")
    parser.add_argument("--output-path", help="Optional path to save embeddings tensor (.pt)")
    parser.add_argument("--dtype", help="Optional torch dtype override for embeddings (e.g., bf16, fp16, fp32)")
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=1,
        help="If >1, run a warmup embedding call on the first N samples before benchmarking",
    )
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)

    from src.tasks.embedding import run_embedding

    model_id = args.model_id or MODEL_ID_MAP.get(args.model, args.model)

    backend = (args.backend or "torch").lower()
    backend_kwargs: Dict[str, Any] = {}
    if backend in {"vllm", "vllm-offline"}:
        backend_kwargs.update(
            {
                "tensor_parallel_size": args.tp_size,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        )
    elif backend in {"sglang-offline", "sglang_offline"}:
        backend_kwargs.update({"tp_size": args.tp_size})

    base_url = _norm_base_url(args.base_url)

    def _maybe_warmup(inputs: List[Any], modality: str) -> None:
        if int(args.warmup_samples) <= 1:
            return
        warmup_n = int(args.warmup_samples)
        warmup_inputs = inputs[: min(warmup_n, len(inputs))]
        if not warmup_inputs:
            return
        print(f"[run_embedding] warmup=true modality={modality} samples={len(warmup_inputs)}")
        _ = run_embedding(
            model_id=model_id,
            backend_name=args.backend,
            inputs=warmup_inputs,
            modality=modality,
            device=args.device,
            base_url=base_url,
            api=args.api,
            api_key=args.api_key,
            timeout=args.timeout,
            image_transport=args.image_transport,
            batch_size=min(args.batch_size, max(1, len(warmup_inputs))),
            max_length=args.max_length,
            normalize=args.normalize,
            encoding_format=args.encoding_format,
            use_amx=args.use_amx,
            dtype=args.dtype,
            **backend_kwargs,
        )

    if args.dataset == "flickr8k":
        from src.data import load_flickr8k

        images_dir = args.flickr8k_images_dir or os.path.dirname(args.dataset_path or "")
        captions_file = args.flickr8k_captions_file or args.dataset_path
        flickr_modality = (args.flickr8k_modality or "both").lower()

        ds = load_flickr8k(
            images_dir=images_dir,
            captions_file=captions_file,
            captions_per_image=args.flickr8k_captions_per_image,
            modality=flickr_modality,  # type: ignore[arg-type]
            max_images=args.max_samples,
        )

        rec: Dict[str, Any] = {
            "dataset": "flickr8k",
            "modality": flickr_modality,
            "captions_per_image": ds.captions_per_image,
            "batch_size": args.batch_size,
            "warmup_samples": int(args.warmup_samples),
        }

        if flickr_modality in {"both", "text"}:
            _maybe_warmup(ds.texts, "text")
            t0 = time.time()
            txt = run_embedding(
                model_id=model_id,
                backend_name=args.backend,
                inputs=ds.texts,
                modality="text",
                device=args.device,
                base_url=base_url,
                api=args.api,
                api_key=args.api_key,
                timeout=args.timeout,
                image_transport=args.image_transport,
                batch_size=args.batch_size,
                max_length=args.max_length,
                normalize=args.normalize,
                encoding_format=args.encoding_format,
                use_amx=args.use_amx,
                dtype=args.dtype,
                **backend_kwargs,
            )
            t1 = time.time()
            elapsed = t1 - t0
            bs = max(1, int(args.batch_size))
            num_batches = int(math.ceil(len(ds.texts) / bs)) if len(ds.texts) > 0 else 0
            rec.update(
                {
                    "text_count": len(ds.texts),
                    "text_time_sec": elapsed,
                    "text_tps": (len(ds.texts) / elapsed) if elapsed > 0 else float("inf"),
                    "text_num_batches": num_batches,
                    "text_avg_batch_time_sec": (elapsed / num_batches) if num_batches > 0 else 0.0,
                    "text_shape": _shape_list(txt),
                }
            )

        if flickr_modality in {"both", "image"}:
            _maybe_warmup(ds.image_paths, "image")
            v0 = time.time()
            img = run_embedding(
                model_id=model_id,
                backend_name=args.backend,
                inputs=ds.image_paths,
                modality="image",
                device=args.device,
                base_url=base_url,
                api=args.api,
                api_key=args.api_key,
                timeout=args.timeout,
                image_transport=args.image_transport,
                batch_size=args.batch_size,
                max_length=args.max_length,
                normalize=args.normalize,
                encoding_format=args.encoding_format,
                use_amx=args.use_amx,
                dtype=args.dtype,
                **backend_kwargs,
            )
            v1 = time.time()
            elapsed = v1 - v0
            bs = max(1, int(args.batch_size))
            num_batches = int(math.ceil(len(ds.image_paths) / bs)) if len(ds.image_paths) > 0 else 0
            rec.update(
                {
                    "image_count": len(ds.image_paths),
                    "image_time_sec": elapsed,
                    "image_tps": (len(ds.image_paths) / elapsed) if elapsed > 0 else float("inf"),
                    "image_num_batches": num_batches,
                    "image_avg_batch_time_sec": (elapsed / num_batches) if num_batches > 0 else 0.0,
                    "image_shape": _shape_list(img),
                }
            )

        print(json.dumps(rec, indent=2, default=str))
        return

    # Default (text-only) datasets
    loader, modality = _get_dataset_loader(args.dataset)

    if args.dataset == "yahoo_answers":
        inputs = loader(path=args.dataset_path, mode=args.yahoo_mode, max_records=args.max_samples)
    else:
        inputs = loader(path=args.dataset_path, max_records=args.max_samples)

    if not inputs:
        raise ValueError(f"No inputs loaded from dataset {args.dataset} at {args.dataset_path}")

    _maybe_warmup(inputs, modality)

    t0 = time.time()
    result = run_embedding(
        model_id=model_id,
        backend_name=args.backend,
        inputs=inputs,
        modality=modality,
        device=args.device,
        base_url=base_url,
        api=args.api,
        api_key=args.api_key,
        timeout=args.timeout,
        image_transport=args.image_transport,
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize=args.normalize,
        encoding_format=args.encoding_format,
        use_amx=args.use_amx,
        dtype=args.dtype,
        **backend_kwargs,
    )
    t1 = time.time()

    count = len(inputs)
    elapsed = t1 - t0
    tps = (count / elapsed) if elapsed > 0 else float("inf")

    bs = max(1, int(args.batch_size))
    num_batches = int(math.ceil(count / bs)) if count > 0 else 0
    avg_batch_time_sec = (elapsed / num_batches) if num_batches > 0 else 0.0

    summary = {
        "count": count,
        "time_sec": elapsed,
        "tps": tps,
        "batch_size": bs,
        "num_batches": num_batches,
        "avg_batch_time_sec": avg_batch_time_sec,
        "shape": _shape_list(result),
    }

    if args.output_path:
        import torch

        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        torch.save(result, args.output_path)
        summary["output_path"] = args.output_path

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # When stdout is closed early (e.g. `| head`), exit quietly.
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.exit(0)
