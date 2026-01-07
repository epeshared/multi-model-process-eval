from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union


def _normalize_batch(
    image_paths: Union[str, Sequence[str]],
    prompt: Union[str, Sequence[str]],
) -> tuple[list[Any], list[str]]:
    imgs = [image_paths] if isinstance(image_paths, str) else list(image_paths)
    pr = [prompt] * len(imgs) if isinstance(prompt, str) else list(prompt)
    if len(pr) != len(imgs):
        raise ValueError("prompt must be a single string or same length as image_paths")
    return imgs, pr


def load_omni_session(
    model_id: str,
    *,
    backend_name: str = "sglang",
    base_url: Optional[str] = None,
    api_key: str = "",
    timeout: float = 600.0,
    image_transport: str = "data-url",
    # vLLM offline knobs
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Load/init a reusable Omni session.

    Rationale: keep Omni-family model/backend quirks isolated from the plain VL task.

    Current scope: image+text chat (audio/video not yet implemented in this repo).
    """

    backend = (backend_name or "sglang").lower()

    if backend in {"sglang", "sglang-http"}:
        from .omni_backends.sglang_http import SGLangHTTPOMNIClient

        if not base_url:
            raise ValueError("base_url is required for backend=sglang")
        sess = SGLangHTTPOMNIClient(
            base_url=base_url,
            model=model_id,
            api_key=api_key,
            timeout=timeout,
            image_transport=image_transport,
        )
        setattr(sess, "_backend_tag", "sglang-http")
        return sess

    if backend in {"vllm-http", "vllm_openai", "vllm-http-openai"}:
        from .omni_backends.vllm_http import VLLMHTTPOMNIClient

        if not base_url:
            raise ValueError("base_url is required for backend=vllm-http")
        return VLLMHTTPOMNIClient(
            base_url=base_url,
            model=model_id,
            api_key=api_key,
            timeout=timeout,
            image_transport=image_transport,
        )

    if backend in {"vllm", "vllm-offline"}:
        from .omni_backends.vllm_offline import VLLMOfflineOMNIClient

        return VLLMOfflineOMNIClient(
            model=model_id,
            dtype=(dtype or "auto"),
            device=(device or "cuda"),
            tensor_parallel_size=int(kwargs.pop("tensor_parallel_size", kwargs.pop("tp_size", 1))),
            max_model_len=kwargs.pop("max_model_len", 8192),
            gpu_memory_utilization=float(kwargs.pop("gpu_memory_utilization", 0.90)),
            **kwargs,
        )

    raise ValueError(f"Unsupported OMNI backend: {backend_name}")


def chat_with_session(
    session: Any,
    *,
    image_paths: Union[str, Sequence[str]],
    prompt: Union[str, Sequence[str]],
    max_new_tokens: int = 128,
    profile: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[str]:
    imgs, pr = _normalize_batch(image_paths, prompt)
    profile_kwargs = dict(profile_kwargs or {})

    if not hasattr(session, "chat"):
        raise ValueError(f"Session does not have chat(): {type(session)}")

    is_sglang_http = getattr(session, "_backend_tag", "") == "sglang-http"

    # profile only applies to sglang-http
    if profile and is_sglang_http:
        started = False
        try:
            if hasattr(session, "start_profile"):
                session.start_profile(**profile_kwargs)
                started = True
            return session.chat(
                image_paths=imgs,
                prompts=pr,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        finally:
            if started and hasattr(session, "stop_profile"):
                try:
                    session.stop_profile(**profile_kwargs)
                except Exception:
                    pass

    return session.chat(image_paths=imgs, prompts=pr, max_new_tokens=max_new_tokens, **kwargs)


def run_omni_chat(
    model_id: str,
    *,
    image_paths: Union[str, Sequence[str]],
    prompt: Union[str, Sequence[str]],
    backend_name: str = "sglang",
    max_new_tokens: int = 128,
    base_url: Optional[str] = None,
    api_key: str = "",
    timeout: float = 600.0,
    image_transport: str = "data-url",
    # vLLM offline knobs
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    profile: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[str]:
    session = load_omni_session(
        model_id,
        backend_name=backend_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        image_transport=image_transport,
        device=device,
        dtype=dtype,
        **kwargs,
    )
    return chat_with_session(
        session,
        image_paths=image_paths,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        profile=profile,
        profile_kwargs=profile_kwargs,
    )
