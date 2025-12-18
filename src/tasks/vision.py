from __future__ import annotations

from typing import Any, List, Optional

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPProcessor,
    OwlViTForObjectDetection,
    OwlViTProcessor,
    pipeline,
)


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def run_image_classification(model_id: str, image_path: str, device: Optional[str] = None) -> Any:
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    if device:
        model.to(device)
    image = load_image(image_path)
    inputs = proc(image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = logits.softmax(dim=-1)[0]
    return [(model.config.id2label[i], scores[i].item()) for i in scores.argsort(descending=True)]


def run_clip_similarity(
    model_id: str,
    image_path: str,
    text_queries: List[str],
    device: Optional[str] = None,
) -> Any:
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    if device:
        model.to(device)
    image = load_image(image_path)
    inputs = processor(text=text_queries, images=image, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logits = text_embeds @ image_embeds.T
    scores = logits[:, 0].tolist()
    return list(zip(text_queries, scores))


def run_owlvit_detection(model_id: str, image_path: str, text_queries: List[str], device: Optional[str] = None) -> Any:
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id)
    if device:
        model.to(device)
    image = load_image(image_path)
    inputs = processor(text=text_queries, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs


def run_image_to_text(model_id: str, image_path: str, device: Optional[str] = None, **kwargs: Any) -> Any:
    text_pipe = pipeline("image-to-text", model=model_id, device=device)
    image = load_image(image_path)
    return text_pipe(image, **kwargs)


def run_image_text_matching(model_id: str, image_path: str, text: str, device: Optional[str] = None) -> Any:
    itm_pipe = pipeline("image-text-to-text", model=model_id, device=device)
    image = load_image(image_path)
    return itm_pipe(image, text)
