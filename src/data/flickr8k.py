from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

__all__ = ["Flickr8kData", "load_flickr8k"]


Flickr8kModality = Literal["both", "text", "image"]


@dataclass(frozen=True)
class Flickr8kData:
    image_paths: List[str]
    texts: List[str]
    captions_per_image: int


def _read_flickr8k_captions(token_txt: str) -> Dict[str, List[str]]:
    """Parse Flickr8k.token.txt -> {filename: [captions...]}"""

    mp: Dict[str, List[str]] = {}
    with open(token_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: <image_name>#<caption_id>\t<caption>
            left, cap = line.split("\t", 1)
            img, _ = left.split("#", 1)
            mp.setdefault(img, []).append(cap)
    return mp


def load_flickr8k(
    *,
    images_dir: str,
    captions_file: str,
    captions_per_image: int = 1,
    modality: Flickr8kModality = "both",
    max_images: int = -1,
) -> Flickr8kData:
    """Load Flickr8k inputs for embedding.

    Args:
        images_dir: Directory containing Flickr8k images (e.g., Flicker8k_Dataset).
        captions_file: Path to Flickr8k.token.txt.
        captions_per_image: How many captions to emit per image (>=1).
        modality: both|text|image.
        max_images: Maximum number of images to use (-1 for all). This also limits
            caption generation proportionally (captions_per_image per image).

    Returns:
        Flickr8kData with:
          - image_paths: list[str] (empty if modality=text)
          - texts: list[str] (empty if modality=image)
    """

    modality_l = (modality or "both").lower().strip()  # type: ignore[assignment]
    if modality_l not in {"both", "text", "image"}:
        raise ValueError("modality must be one of: both|text|image")

    per_img = max(1, int(captions_per_image))

    if not captions_file or not os.path.exists(captions_file):
        raise FileNotFoundError(f"Flickr8k captions file not found: {captions_file}")

    cap_map = _read_flickr8k_captions(captions_file)
    if not cap_map:
        raise RuntimeError("Flickr8k captions are empty")

    image_paths: List[str] = []
    texts: List[str] = []

    for fn, caps in sorted(cap_map.items()):
        # If we are embedding images, require the image file to exist.
        if modality_l in {"both", "image"}:
            p = os.path.join(images_dir, fn)
            if not os.path.exists(p):
                # Skip missing images so we don't send bad paths.
                continue
            image_paths.append(p)

        if modality_l in {"both", "text"}:
            if not caps:
                take = [""] * per_img
            else:
                take = caps[:per_img] if len(caps) >= per_img else (caps + caps[: (per_img - len(caps))])
            texts.extend(take)

        if max_images and max_images > 0:
            if modality_l in {"both", "image"}:
                if len(image_paths) >= max_images:
                    break
            else:
                # text-only: count images implied by captions_per_image
                if (len(texts) // per_img) >= max_images:
                    break

    # Validate outputs.
    if modality_l in {"both", "image"} and not image_paths:
        raise RuntimeError("No Flickr8k images found matching captions file")
    if modality_l in {"both", "text"} and not texts:
        raise RuntimeError("No Flickr8k captions found")

    # If modality=image, ensure we don't accidentally return captions.
    if modality_l == "image":
        texts = []
    if modality_l == "text":
        image_paths = []

    return Flickr8kData(image_paths=image_paths, texts=texts, captions_per_image=per_img)
