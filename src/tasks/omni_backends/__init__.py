from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SGLangHTTPOMNIClient",
    "VLLMHTTPOMNIClient",
    "VLLMOfflineOMNIClient",
]

_SYMBOL_TO_MODULE = {
    "SGLangHTTPOMNIClient": ".sglang_http",
    "VLLMHTTPOMNIClient": ".vllm_http",
    "VLLMOfflineOMNIClient": ".vllm_offline",
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    mod = _SYMBOL_TO_MODULE.get(name)
    if not mod:
        raise AttributeError(name)
    m = importlib.import_module(mod, __name__)
    return getattr(m, name)
