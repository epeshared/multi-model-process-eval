from __future__ import annotations

from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def run_vision_language_chat(
    model_id: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 128,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16)
    if device:
        model.to(device)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)
