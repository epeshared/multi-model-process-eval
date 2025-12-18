from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from .registry import MODEL_REGISTRY, available_models, run_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-backend model runner")
    parser.add_argument("--model-key", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--backend", default="torch")
    parser.add_argument("--prompt", help="Prompt for generation or VQA tasks")
    parser.add_argument("--text", help="Plain text input")
    parser.add_argument("--texts", nargs="*", help="Multiple text inputs")
    parser.add_argument("--image", help="Path to an input image")
    parser.add_argument("--audio", help="Path to an input audio file")
    parser.add_argument("--video", help="Path to an input video file")
    parser.add_argument("--device", help="Device id or name, e.g., cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser


def parse_args(argv: Any = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return vars(args)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    model_key = args.pop("model_key")
    backend = args.pop("backend")
    result = run_model(model_key=model_key, backend=backend, **args)
    try:
        print(json.dumps(result, indent=2, default=str))
    except TypeError:
        print(result)


if __name__ == "__main__":
    main()
