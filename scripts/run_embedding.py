#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, List

from src.registry import run_model

EMBED_MODELS: List[str] = [
    "qwen3-embedding-4b",
    "qwen3-embedding-0.6b",
]

EMBED_BACKENDS: List[str] = [
    "torch",
    "sglang",
    "sglang-offline",
    "vllm",
    "vllm-http",
]


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding entry point")
    parser.add_argument("--model", required=True, choices=EMBED_MODELS)
    parser.add_argument("--backend", choices=EMBED_BACKENDS, default="torch")
    parser.add_argument("--texts", nargs="+", required=True, help="Texts to embed")
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
    parser.add_argument("--encoding-format", help="Encoding format for vLLM HTTP (e.g., bf16)")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for vLLM offline")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max model length for vLLM offline")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilization for vLLM offline")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    result = run_model(
        model_key=args.model,
        backend=args.backend,
        texts=args.texts,
        device=args.device,
        base_url=args.base_url,
        api=args.api,
        api_key=args.api_key,
        timeout=args.timeout,
        image_transport=args.image_transport,
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize=args.normalize,
        encoding_format=args.encoding_format,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(json.dumps(result.tolist(), indent=2))


if __name__ == "__main__":
    main()
