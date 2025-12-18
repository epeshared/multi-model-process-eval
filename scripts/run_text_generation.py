#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, List

from src.registry import run_model

TEXT_MODELS: List[str] = [
    "qwen3-1.7b",
    "qwen2.5-omni-7b",
    "flan-t5-summarization",
    "pythia-6.9b",
]


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text generation entry point")
    parser.add_argument("--model", required=True, choices=TEXT_MODELS)
    parser.add_argument("--backend", default="torch", choices=["torch", "vllm", "sglang"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    result = run_model(
        model_key=args.model,
        backend=args.backend,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
