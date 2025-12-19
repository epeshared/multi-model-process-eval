#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from src.tasks.text_classification import run_text_classification

CLASSIFICATION_MODELS: List[str] = [
    "klue-roberta-intent",
    "financial-sentiment",
    "topic-classification",
]

MODEL_ID_MAP: Dict[str, str] = {
    "klue-roberta-intent": "bespin-global/klue-roberta-small-3i4k-intent-classification",
    "financial-sentiment": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "topic-classification": "dstefa/roberta-base_topic_classification_nyt_news",
}


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text classification entry point")
    parser.add_argument("--model", required=True, choices=CLASSIFICATION_MODELS)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    model_id = MODEL_ID_MAP.get(args.model, args.model)
    result = run_text_classification(model_id=model_id, text=args.text, device=args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
