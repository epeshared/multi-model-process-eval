from __future__ import annotations

import glob
import json
import os
from typing import Any, List, Optional

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _limit(items: List[Any], max_samples: int) -> List[Any]:
    if max_samples is not None and max_samples > 0:
        return items[:max_samples]
    return items


def _dedup_preserve_order(items: List[Any]) -> List[Any]:
    seen = set()
    deduped: List[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _load_texts_from_file(path: str, max_samples: int) -> List[str]:
    texts: List[str] = []
    if not path:
        return texts
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                texts.append(line)
            if max_samples > 0 and len(texts) >= max_samples:
                break
    return texts


def _load_texts_from_jsonl(path: str, field: str, max_samples: int) -> List[str]:
    texts: List[str] = []
    if not path:
        return texts
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            val = obj.get(field)
            if val is None:
                continue
            if isinstance(val, list):
                texts.extend([str(v) for v in val if v is not None])
            else:
                texts.append(str(val))
            if max_samples > 0 and len(texts) >= max_samples:
                break
    return texts


def _load_image_paths(image: Optional[str], image_dir: Optional[str], image_glob: Optional[str], max_samples: int) -> List[str]:
    paths: List[str] = []
    exts = _IMAGE_EXTS

    if image:
        paths.append(image)

    if image_glob:
        paths.extend(glob.glob(image_glob, recursive=True))

    if image_dir:
        for dirpath, _, filenames in os.walk(image_dir):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                paths.append(full)

    filtered: List[str] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if exts and ext not in exts:
            continue
        filtered.append(p)
        if max_samples > 0 and len(filtered) >= max_samples:
            break

    return filtered


def load_embedding_inputs(
    *,
    modality: str = "text",
    texts: Optional[List[str]] = None,
    text_file: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    jsonl_field: str = "text",
    image: Optional[str] = None,
    image_dir: Optional[str] = None,
    image_glob: Optional[str] = None,
    max_samples: int = -1,
) -> List[Any]:
    """Load embedding inputs from multiple sources.

    Args:
        modality: "text" or "image".
        texts: Inline text inputs (highest priority when modality=text).
        text_file: File with one text per line.
        jsonl_path: JSONL file containing records with the specified field.
        jsonl_field: Field name to extract from each JSON line.
        image: Single image path (modality=image).
        image_dir: Directory to scan recursively for images.
        image_glob: Glob pattern for images (supports **).
        max_samples: Max number of items to keep; -1 keeps all.

    Returns:
        A list of inputs ready for embedding.
    """

    mode = (modality or "text").lower()

    if mode == "text":
        collected: List[str] = []
        if texts:
            collected.extend([str(t) for t in texts if t is not None])
        if text_file:
            collected.extend(_load_texts_from_file(text_file, max_samples))
        if jsonl_path:
            collected.extend(_load_texts_from_jsonl(jsonl_path, jsonl_field, max_samples))
        collected = _dedup_preserve_order(collected)
        return _limit(collected, max_samples)

    if mode == "image":
        images = _load_image_paths(image, image_dir, image_glob, max_samples)
        images = _dedup_preserve_order(images)
        return _limit(images, max_samples)

    raise ValueError(f"Unsupported modality: {modality}")
