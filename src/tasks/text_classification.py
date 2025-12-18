from __future__ import annotations

from typing import Any, Optional

from transformers import pipeline


def run_text_classification(model_id: str, text: str, device: Optional[str] = None, **kwargs: Any) -> Any:
    clf = pipeline("text-classification", model=model_id, device=device)
    return clf(text, **kwargs)
