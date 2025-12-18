#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from src.registry import run_model

VIDEO_MODELS = ["video-blip-ego4d"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Video caption/classification entry point")
    parser.add_argument("--model", required=True, choices=VIDEO_MODELS)
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_model(model_key=args.model, backend="torch", video=args.video, device=args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
