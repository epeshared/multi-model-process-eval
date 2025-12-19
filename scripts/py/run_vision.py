#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from src.tasks.vision import (
    run_clip_similarity,
    run_image_classification,
    run_image_text_matching,
    run_image_to_text,
    run_owlvit_detection,
)

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

MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "openai-clip-vit-base-patch32": {
        "model_id": "openai/clip-vit-base-patch32",
        "task": "clip-similarity",
    },
    "nsfw-image-detection": {
        "model_id": "nsfw_image_detection",
        "task": "image-classification",
    },
    "aesthetics-predictor-v1": {
        "model_id": "aesthetics-predictor-v1-vit-base-patch16",
        "task": "image-classification",
    },
    "aesthetics-predictor-v2": {
        "model_id": "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
        "task": "image-classification",
    },
    "watermark-detector": {
        "model_id": "amrul-hzz/watermark_detector",
        "task": "image-classification",
    },
    "blip2-opt-2.7b": {
        "model_id": "Salesforce/blip2-opt-2.7b",
        "task": "image-to-text",
    },
    "owlvit-base": {
        "model_id": "google/owlvit-base-patch32",
        "task": "object-detection",
    },
    "blip-itm-base": {
        "model_id": "Salesforce/blip-itm-base-coco",
        "task": "image-text-matching",
    },
}


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
    info = MODEL_INFO.get(args.model, {"model_id": args.model, "task": "image-classification"})
    model_id = info.get("model_id", args.model)
    task = info.get("task", "image-classification")

    texts_list: List[str] = args.texts or ([args.text] if args.text else [])
    result: Any

    if task == "clip-similarity":
        if not texts_list:
            raise ValueError("CLIP similarity requires --text or --texts")
        result = run_clip_similarity(model_id=model_id, image_path=args.image, text_queries=texts_list, device=args.device)
    elif task == "object-detection":
        if not texts_list:
            raise ValueError("Object detection requires --text or --texts")
        result = run_owlvit_detection(model_id=model_id, image_path=args.image, text_queries=texts_list, device=args.device)
    elif task == "image-text-matching":
        if not args.text:
            raise ValueError("Image-text matching requires --text")
        result = run_image_text_matching(model_id=model_id, image_path=args.image, text=args.text, device=args.device)
    elif task == "image-to-text":
        result = run_image_to_text(model_id=model_id, image_path=args.image, device=args.device)
    else:  # image-classification
        result = run_image_classification(model_id=model_id, image_path=args.image, device=args.device)

    try:
        print(json.dumps(result, indent=2, default=str))
    except TypeError:
        print(result)


if __name__ == "__main__":
    main()
