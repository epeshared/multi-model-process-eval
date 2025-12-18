#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from src.registry import run_model

TRANSLATION_MODELS = ["opus-mt-zh-en", "opus-mt-en-zh"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Translation entry point")
    parser.add_argument("--model", required=True, choices=TRANSLATION_MODELS)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_model(model_key=args.model, backend="torch", text=args.text, device=args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
