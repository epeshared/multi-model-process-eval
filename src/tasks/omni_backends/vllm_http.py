from __future__ import annotations

from typing import Any

from ..vl_backends.vllm_http import VLLMHTTPVLClient


class VLLMHTTPOMNIClient(VLLMHTTPVLClient):
    """vLLM OpenAI-compatible HTTP backend for Omni models.

    Reuses the existing VL HTTP client implementation.
    """

    name = "vllm-http-omni"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
