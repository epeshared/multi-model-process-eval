from __future__ import annotations

from typing import Any, Optional

from transformers import pipeline


def run_audio_classification(model_id: str, audio_path: str, device: Optional[str] = None, **kwargs: Any) -> Any:
    audio_pipe = pipeline("audio-classification", model=model_id, device=device)
    return audio_pipe(audio_path, **kwargs)


def run_speech_recognition(model_id: str, audio_path: str, device: Optional[str] = None, **kwargs: Any) -> Any:
    asr_pipe = pipeline("automatic-speech-recognition", model=model_id, device=device)
    return asr_pipe(audio_path, **kwargs)
