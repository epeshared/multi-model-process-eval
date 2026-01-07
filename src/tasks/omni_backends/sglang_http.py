from __future__ import annotations

from typing import Any

from ..vl_backends.sglang_http import SGLangHTTPVLClient


class SGLangHTTPOMNIClient(SGLangHTTPVLClient):
    """SGLang HTTP backend for Omni models.

    For now we reuse the OpenAI-compatible VL chat schema:
    POST {base_url}/v1/chat/completions with a message containing image_url + text.

    This lives under omni_backends to keep model-family specific logic isolated
    from the plain VL task.
    """

    name = "sglang-http-omni"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        setattr(self, "_backend_tag", "sglang-http")
