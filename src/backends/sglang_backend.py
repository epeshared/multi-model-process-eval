from __future__ import annotations

import importlib
from typing import Any, Dict

from .base import Backend, MissingDependency


class SGLangBackend(Backend):
    name = "sglang"

    def __init__(self) -> None:
        if not importlib.util.find_spec("sglang"):
            raise MissingDependency("sglang is required for the sglang backend")
        import sglang as sgl  # type: ignore

        self.sgl = sgl

    def load(self, model_id: str, **kwargs: Any) -> Any:
        return self.sgl.Runtime(model_path=model_id, **kwargs)

    def infer(self, model: Any, inputs: Dict[str, Any], **kwargs: Any) -> Any:
        prompt: str = inputs.get("prompt", "")
        max_new_tokens: int = inputs.get("max_new_tokens", 128)
        sgl = self.sgl

        @sgl.function
        def chat_fn(s):
            s += sgl.user(prompt)
            s += sgl.assistant(max_new_tokens=max_new_tokens)

        return chat_fn.run(rt=model)

    def supports(self, task: str) -> bool:
        return task in {"text-generation", "chat"}
