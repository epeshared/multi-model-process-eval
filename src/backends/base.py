from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Backend(ABC):
    """Abstract backend interface for model loading and inference."""

    name: str

    @abstractmethod
    def load(self, model_id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def infer(self, model: Any, inputs: Dict[str, Any], **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def supports(self, task: str) -> bool:
        raise NotImplementedError


class MissingDependency(RuntimeError):
    pass
