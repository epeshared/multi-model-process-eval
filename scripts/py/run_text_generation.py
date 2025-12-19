#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from src.tasks.text_generation import run_text_generation

TEXT_MODELS: List[str] = [
    "qwen3-1.7b",
    "qwen2.5-omni-7b",
    "flan-t5-summarization",
    "pythia-6.9b",
]

MODEL_ID_MAP: Dict[str, str] = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",
    "flan-t5-summarization": "mrm8488/flan-t5-large-finetuned-openai-summarize_from_feedback",
    "pythia-6.9b": "EleutherAI/pythia-6.9b-deduped",
}


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
    model_id = MODEL_ID_MAP.get(args.model, args.model)
    result = run_text_generation(
        model_id=model_id,
        backend_name=args.backend,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
