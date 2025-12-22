from __future__ import annotations

from typing import Any, Optional

from .vl import run_vl_chat


def run_vision_language_chat(
    model_id: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 128,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    # Keep API compatibility but route implementation through the VL task.
    # Allow passing VL-specific options via kwargs without changing the signature.
    dtype = kwargs.pop("dtype", None)
    trust_remote_code = bool(kwargs.pop("trust_remote_code", False))
    attn_implementation = kwargs.pop("attn_implementation", None)

    return run_vl_chat(
        model_id=model_id,
        image_paths=image_path,
        prompt=prompt,
        backend_name="torch",
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        **kwargs,
    )
