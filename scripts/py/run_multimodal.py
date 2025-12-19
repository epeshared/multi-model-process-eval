#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from src.tasks.multimodal import run_vision_language_chat

MULTIMODAL_MODELS: List[str] = [
    "qwen2.5-vl-7b-instruct",
    "llava-vicuna-7b",
]

MODEL_ID_MAP: Dict[str, str] = {
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava-vicuna-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
}


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
    model_id = MODEL_ID_MAP.get(args.model, args.model)
    result = run_vision_language_chat(
        model_id=model_id,
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
