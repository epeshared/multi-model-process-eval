#!/usr/bin/env python
from __future__ import annotations

import argparse

from typing import Callable, Dict

from src.tasks.diffusion import run_stable_diffusion_v1, run_stable_diffusion_xl

DIFFUSION_MODELS = ["stable-diffusion-v1-4", "stable-diffusion-xl"]

MODEL_INFO: Dict[str, Dict[str, Callable]] = {
    "stable-diffusion-v1-4": {
        "model_id": "CompVis/stable-diffusion-v1-4",
        "runner": run_stable_diffusion_v1,
    },
    "stable-diffusion-xl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "runner": run_stable_diffusion_xl,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Diffusion image generation")
    parser.add_argument("--model", required=True, choices=DIFFUSION_MODELS)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    info = MODEL_INFO.get(args.model, {})
    runner = info.get("runner", run_stable_diffusion_v1)
    model_id = info.get("model_id", args.model)

    images = runner(
        model_id=model_id,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
    )
    for idx, img in enumerate(images):
        out_path = f"{args.model.replace('-', '_')}_{idx}.png"
        img.save(out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
