#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, List

from src.registry import get_spec, run_model

VISION_MODELS: List[str] = [
    "openai-clip-vit-base-patch32",
    "nsfw-image-detection",
    "aesthetics-predictor-v1",
    "aesthetics-predictor-v2",
    "watermark-detector",
    "blip2-opt-2.7b",
    "owlvit-base",
    "blip-itm-base",
]


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision entry point")
    parser.add_argument("--model", required=True, choices=VISION_MODELS)
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--text", help="Single text input for matching tasks")
    parser.add_argument("--texts", nargs="*", help="Multiple text inputs for CLIP or detection")
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    spec = get_spec(args.model)
    kwargs: dict = {"image": args.image, "device": args.device}

    if spec.task == "clip-similarity":
        kwargs["texts"] = args.texts or ([args.text] if args.text else [])
    elif spec.task == "image-text-matching":
        kwargs["text"] = args.text or ""
    elif spec.task == "object-detection":
        kwargs["texts"] = args.texts or ([args.text] if args.text else [])

    result = run_model(model_key=args.model, backend="torch", **kwargs)
    try:
        print(json.dumps(result, indent=2, default=str))
    except TypeError:
        print(result)


if __name__ == "__main__":
    main()
