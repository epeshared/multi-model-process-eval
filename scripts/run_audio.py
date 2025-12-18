#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, List

from src.registry import get_spec, run_model

AUDIO_MODELS: List[str] = ["qwen-audio", "ast-audioset"]


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio entry point")
    parser.add_argument("--model", required=True, choices=AUDIO_MODELS)
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    spec = get_spec(args.model)
    result = run_model(model_key=args.model, backend="torch", audio=args.audio, device=args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
