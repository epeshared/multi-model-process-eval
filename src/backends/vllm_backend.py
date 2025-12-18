from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from .base import Backend, MissingDependency


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self) -> None:
        if not importlib.util.find_spec("vllm"):
            raise MissingDependency("vllm is required for the vllm backend")
        from vllm import LLM  # type: ignore

        self._LLM = LLM

    def load(self, model_id: str, **kwargs: Any) -> Any:
        return self._LLM(model=model_id, **kwargs)

    def infer(self, model: Any, inputs: Dict[str, Any], **kwargs: Any) -> Any:
        from vllm import SamplingParams  # type: ignore

        prompts: List[str] = inputs.get("prompts", [])
        max_new_tokens: int = inputs.get("max_new_tokens", 128)
        sampling_params = SamplingParams(max_tokens=max_new_tokens, **kwargs)
        return model.generate(prompts, sampling_params)

    def supports(self, task: str) -> bool:
        return task in {"text-generation", "chat"}
