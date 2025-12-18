from __future__ import annotations

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline


def run_stable_diffusion_v1(
    model_id: str,
    prompt: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    device: Optional[str] = None,
    **kwargs: Any,
):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    if device:
        pipe = pipe.to(device)
    return pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, **kwargs).images


def run_stable_diffusion_xl(
    model_id: str,
    prompt: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.0,
    device: Optional[str] = None,
    **kwargs: Any,
):
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    if device:
        pipe = pipe.to(device)
    return pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, **kwargs).images


def run_general_diffusion(
    model_id: str,
    prompt: str,
    num_inference_steps: int = 30,
    device: Optional[str] = None,
    **kwargs: Any,
):
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    if device:
        pipe = pipe.to(device)
    return pipe(prompt=prompt, num_inference_steps=num_inference_steps, **kwargs).images
