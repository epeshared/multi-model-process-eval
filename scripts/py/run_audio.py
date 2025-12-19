#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from src.tasks.audio import run_audio_classification, run_speech_recognition

AUDIO_MODELS: List[str] = ["qwen-audio", "ast-audioset"]

MODEL_INFO: Dict[str, Dict[str, str]] = {
    "qwen-audio": {"model_id": "Qwen/Qwen-Audio", "task": "speech-recognition"},
    "ast-audioset": {"model_id": "MIT/ast-finetuned-audioset-10-10-0.4593", "task": "audio-classification"},
}


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio entry point")
    parser.add_argument("--model", required=True, choices=AUDIO_MODELS)
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    info = MODEL_INFO.get(args.model, {"model_id": args.model, "task": "audio-classification"})
    model_id = info["model_id"]
    task = info.get("task", "audio-classification")

    if task == "speech-recognition":
        result = run_speech_recognition(model_id=model_id, audio_path=args.audio, device=args.device)
    else:
        result = run_audio_classification(model_id=model_id, audio_path=args.audio, device=args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
