from __future__ import annotations

from typing import Any, Optional

from transformers import pipeline


def run_translation(model_id: str, text: str, device: Optional[str] = None, **kwargs: Any) -> Any:
    trans_pipe = pipeline("translation", model=model_id, device=device)
    return trans_pipe(text, **kwargs)
