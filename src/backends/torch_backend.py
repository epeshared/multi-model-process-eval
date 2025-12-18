from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from .base import Backend, MissingDependency


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        if not importlib.util.find_spec("torch"):
            raise MissingDependency("torch is required for the torch backend")
        if not importlib.util.find_spec("transformers"):
            raise MissingDependency("transformers is required for the torch backend")

    def load(self, model_id: str, **kwargs: Any) -> Any:
        from transformers import pipeline

        task = kwargs.pop("task", None)
        if task is None:
            raise ValueError("'task' must be provided to load a pipeline")
        device = self.device if self.device is not None else kwargs.pop("device", None)
        return pipeline(task=task, model=model_id, device=device, **kwargs)

    def infer(self, model: Any, inputs: Dict[str, Any], **kwargs: Any) -> Any:
        return model(**inputs)

    def supports(self, task: str) -> bool:
        return True
