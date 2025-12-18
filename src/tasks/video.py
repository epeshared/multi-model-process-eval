from __future__ import annotations

from typing import Any, Optional

from transformers import pipeline


def run_video_caption(model_id: str, video_path: str, device: Optional[str] = None, **kwargs: Any) -> Any:
    caption_pipe = pipeline("video-classification", model=model_id, device=device)
    return caption_pipe(video_path, **kwargs)
