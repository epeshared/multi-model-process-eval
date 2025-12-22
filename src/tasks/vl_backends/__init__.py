from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "TorchVisionLanguageClient",
    "SGLangHTTPVLClient",
    "SGLangOfflineVLClient",
    "VLLMHTTPVLClient",
    "VLLMOfflineVLClient",
]


_SYMBOL_TO_MODULE = {
    "TorchVisionLanguageClient": ".torch_vl",
    "SGLangHTTPVLClient": ".sglang_http",
    "SGLangOfflineVLClient": ".sglang_offline",
    "VLLMHTTPVLClient": ".vllm_http",
    "VLLMOfflineVLClient": ".vllm_offline",
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    mod = _SYMBOL_TO_MODULE.get(name)
    if not mod:
        raise AttributeError(name)
    m = importlib.import_module(mod, __name__)
    return getattr(m, name)
