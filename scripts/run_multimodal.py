#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, List

from src.registry import run_model

MULTIMODAL_MODELS: List[str] = [
    "qwen2.5-vl-7b-instruct",
    "llava-vicuna-7b",
]


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision-language entry point")
    parser.add_argument("--model", required=True, choices=MULTIMODAL_MODELS)
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--prompt", required=True, help="Prompt to ask about the image")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    result = run_model(
        model_key=args.model,
        backend="torch",
        image=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
