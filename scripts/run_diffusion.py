#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.registry import run_model

DIFFUSION_MODELS = ["stable-diffusion-v1-4", "stable-diffusion-xl"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Diffusion image generation")
    parser.add_argument("--model", required=True, choices=DIFFUSION_MODELS)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--device", help="Device id, e.g., cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    images = run_model(
        model_key=args.model,
        backend="torch",
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
