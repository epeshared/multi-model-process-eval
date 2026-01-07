from __future__ import annotations

from ..vl_backends.vllm_offline import VLLMOfflineVLClient


class VLLMOfflineOMNIClient(VLLMOfflineVLClient):
    """Local vLLM backend for Omni models.

    Currently Omni image+text can reuse the same multimodal request schema.
    """

    name = "vllm-offline-omni"
