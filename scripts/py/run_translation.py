#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Dict

from src.tasks.translation import run_translation

TRANSLATION_MODELS = ["opus-mt-zh-en", "opus-mt-en-zh"]

MODEL_ID_MAP: Dict[str, str] = {
    "opus-mt-zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "opus-mt-en-zh": "Helsinki-NLP/opus-mt-en-zh",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Translation entry point")
    parser.add_argument("--model", required=True, choices=TRANSLATION_MODELS)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_id = MODEL_ID_MAP.get(args.model, args.model)
    result = run_translation(model_id=model_id, text=args.text, device=args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
