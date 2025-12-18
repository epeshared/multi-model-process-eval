from __future__ import annotations

from typing import Any, Dict, Optional

from ..backends.sglang_backend import SGLangBackend
from ..backends.torch_backend import TorchBackend
from ..backends.vllm_backend import VLLMBackend


def run_text_generation(
    model_id: str,
    backend_name: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Dispatch text generation to the requested backend."""

    if backend_name == "torch":
        backend = TorchBackend(device=device)
        pipe = backend.load(model_id=model_id, task="text-generation", **kwargs)
        return pipe(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    if backend_name == "vllm":
        backend = VLLMBackend()
        model = backend.load(model_id=model_id, **kwargs)
        outputs = backend.infer(
            model,
            {"prompts": [prompt], "max_new_tokens": max_new_tokens},
            temperature=temperature,
        )
        return outputs

    if backend_name == "sglang":
        backend = SGLangBackend()
        runtime = backend.load(model_id=model_id, **kwargs)
        return backend.infer(runtime, {"prompt": prompt, "max_new_tokens": max_new_tokens})

    raise ValueError(f"Unsupported backend: {backend_name}")
