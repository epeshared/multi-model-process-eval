from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union


def _normalize_batch(prompt: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(prompt, str):
        return [prompt]
    return [str(x) for x in list(prompt)]


def load_qwen3_session(
    model_id: str,
    *,
    backend_name: str = "vllm-http",
    base_url: Optional[str] = None,
    api_key: str = "",
    timeout: float = 600.0,
    served_model: str = "",
    **kwargs: Any,
) -> Any:
    """Load/init a reusable Qwen3 (text-only LLM) session.

    For HTTP servers (SGLang / vLLM OpenAI-compatible), the `model` field is
    typically the served model name, not the local HF path.
    """

    backend = (backend_name or "vllm-http").lower()

    http_model = str(served_model or "" or model_id)

    if backend in {"sglang", "sglang-http"}:
        from .qwen3_backends.sglang_http import Qwen3SGLangHTTPClient

        if not base_url:
            raise ValueError("base_url is required for backend=sglang")
        sess = Qwen3SGLangHTTPClient(
            base_url=base_url,
            model=http_model,
            api_key=api_key,
            timeout=float(timeout),
        )
        setattr(sess, "_backend_tag", "sglang-http")
        return sess

    if backend in {"vllm-http", "vllm_openai", "vllm-http-openai"}:
        from .qwen3_backends.vllm_http import Qwen3VLLMHTTPClient

        if not base_url:
            raise ValueError("base_url is required for backend=vllm-http")
        return Qwen3VLLMHTTPClient(
            base_url=base_url,
            model=http_model,
            api_key=api_key,
            timeout=float(timeout),
        )

    raise ValueError(f"Unsupported Qwen3 backend: {backend_name}")


def chat_with_session(
    session: Any,
    *,
    prompt: Union[str, Sequence[str]],
    max_new_tokens: int = 128,
    **kwargs: Any,
) -> List[str]:
    prompts = _normalize_batch(prompt)

    if not hasattr(session, "chat"):
        raise ValueError(f"Session does not have chat(): {type(session)}")

    return session.chat(prompts=prompts, max_new_tokens=int(max_new_tokens), **kwargs)


def run_qwen3_chat(
    model_id: str,
    *,
    prompt: Union[str, Sequence[str]],
    backend_name: str = "vllm-http",
    max_new_tokens: int = 128,
    base_url: Optional[str] = None,
    api_key: str = "",
    timeout: float = 600.0,
    served_model: str = "",
    **kwargs: Any,
) -> List[str]:
    session = load_qwen3_session(
        model_id,
        backend_name=backend_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        served_model=served_model,
        **kwargs,
    )
    return chat_with_session(session, prompt=prompt, max_new_tokens=max_new_tokens, **kwargs)
