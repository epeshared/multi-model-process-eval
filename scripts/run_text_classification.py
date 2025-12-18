#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, List

from src.registry import run_model

CLASSIFICATION_MODELS: List[str] = [
    "klue-roberta-intent",
    "financial-sentiment",
    "topic-classification",
]


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text classification entry point")
    parser.add_argument("--model", required=True, choices=CLASSIFICATION_MODELS)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    result = run_model(model_key=args.model, backend="torch", text=args.text, device=args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
