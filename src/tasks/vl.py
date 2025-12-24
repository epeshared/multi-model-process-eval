# src/tasks/vl.py
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


def load_vl_session(
    model_id: str,
    *,
    backend_name: str = "torch",
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    use_amx: bool = False,
    print_model_info: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    base_url: Optional[str] = None,
    api: str = "v1",
    api_key: str = "",
    timeout: float = 600.0,
    image_transport: str = "data-url",
    **kwargs: Any,
) -> Any:
    """Load/init a reusable VL session.

    This separates expensive backend initialization (model load / client init)
    from per-call chat. Intended for benchmarking where load time should be
    excluded from chat timing.
    """
    backend = (backend_name or "torch").lower()

    if backend == "torch":
        from .vl_backends.torch_vl import TorchVisionLanguageClient

        torch_client = TorchVisionLanguageClient()
        runtime = torch_client.load(
            model_id=model_id,
            device=device,
            dtype=dtype,
            use_amx=use_amx,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        if print_model_info:
            try:
                model = getattr(runtime, "model", None)
                proc = getattr(runtime, "processor", None)
                n_params = None
                if model is not None and hasattr(model, "parameters"):
                    n_params = sum(int(p.numel()) for p in model.parameters())
                cfg = getattr(model, "config", None)
                model_type = getattr(cfg, "model_type", None) if cfg is not None else None
                print(
                    "[vl.load] backend=torch "
                    f"model_id={model_id} model_class={type(model).__name__ if model is not None else None} "
                    f"processor_class={type(proc).__name__ if proc is not None else None} "
                    f"device={getattr(model, 'device', None)} dtype={getattr(model, 'dtype', None)} "
                    f"model_type={model_type} n_params={n_params}"
                )
            except Exception as e:
                print(f"[vl.load] print_model_info failed: {e}")
        return ("torch", torch_client, runtime)

    if backend == "sglang":
        from .vl_backends.sglang_http import SGLangHTTPVLClient

        if not base_url:
            raise ValueError("base_url is required for backend=sglang")
        _ = api  # reserved for future (native/openai toggles)
        if print_model_info:
            print(f"[vl.load] backend=sglang-http base_url={base_url} model_id={model_id}")

        sess = SGLangHTTPVLClient(
            base_url=base_url,
            model=model_id,
            api_key=api_key,
            timeout=timeout,
            image_transport=image_transport,
        )
        setattr(sess, "_backend_tag", "sglang-http")
        return sess

    if backend in {"sglang-offline", "sglang_offline"}:
        from .vl_backends.sglang_offline import SGLangOfflineVLClient

        if print_model_info:
            print(
                f"[vl.load] backend=sglang-offline model_id={model_id} "
                f"device={device or 'cuda'} dtype={dtype or 'auto'}"
            )

        dev = (device or "cuda").lower()
        default_enable_torch_compile = not dev.startswith("cpu")
        default_torch_compile_max_bs = 32 if default_enable_torch_compile else 1
        return SGLangOfflineVLClient(
            model=model_id,
            dtype=(dtype or "auto"),
            device=(device or "cuda"),
            tp_size=int(kwargs.pop("tp_size", kwargs.pop("tensor_parallel_size", 1))),
            dp_size=int(kwargs.pop("dp_size", 1)),
            trust_remote_code=trust_remote_code,
            quantization=kwargs.pop("quantization", None),
            revision=kwargs.pop("revision", None),
            attention_backend=kwargs.pop("attention_backend", None),
            enable_torch_compile=bool(kwargs.pop("enable_torch_compile", default_enable_torch_compile)),
            torch_compile_max_bs=int(kwargs.pop("torch_compile_max_bs", default_torch_compile_max_bs)),
            **kwargs,
        )

    if backend in {"vllm-http", "vllm_openai", "vllm-http-openai"}:
        from .vl_backends.vllm_http import VLLMHTTPVLClient

        if not base_url:
            raise ValueError("base_url is required for backend=vllm-http")
        if print_model_info:
            print(f"[vl.load] backend=vllm-http base_url={base_url} model_id={model_id}")
        return VLLMHTTPVLClient(
            base_url=base_url,
            model=model_id,
            api_key=api_key,
            timeout=timeout,
            image_transport=image_transport,
        )

    if backend in {"vllm", "vllm-offline"}:
        from .vl_backends.vllm_offline import VLLMOfflineVLClient

        if print_model_info:
            print(f"[vl.load] backend=vllm-offline model_id={model_id} device={device or 'cuda'} dtype={dtype or 'auto'}")
        return VLLMOfflineVLClient(
            model=model_id,
            dtype=(dtype or "auto"),
            device=(device or "cuda"),
            tensor_parallel_size=int(kwargs.pop("tensor_parallel_size", kwargs.pop("tp_size", 1))),
            max_model_len=kwargs.pop("max_model_len", 8192),
            gpu_memory_utilization=float(kwargs.pop("gpu_memory_utilization", 0.90)),
            **kwargs,
        )

    raise ValueError(f"Unsupported VL backend: {backend_name}")


def chat_with_session(
    session: Any,
    *,
    image_paths: Union[str, Sequence[str]],
    prompt: Union[str, Sequence[str]],
    max_new_tokens: int = 128,
    # profile knobs (ONLY for sglang-http)
    profile: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[str]:
    imgs, pr = _normalize_batch(image_paths, prompt)
    profile_kwargs = dict(profile_kwargs or {})

    # torch session tuple
    if isinstance(session, tuple) and len(session) == 3 and session[0] == "torch":
        _, torch_client, runtime = session
        # torch backend ignores profile args (keep stable)
        return torch_client.chat(runtime, image_paths=imgs, prompts=pr, max_new_tokens=max_new_tokens, **kwargs)

    if not hasattr(session, "chat"):
        raise ValueError(f"Session does not have chat(): {type(session)}")


    is_sglang_http = getattr(session, "_backend_tag", "") == "sglang-http"

    # profile only applies to sglang-http
    if profile and is_sglang_http:
        started = False
        try:
            # 1) start profile (server-side)
            if hasattr(session, "start_profile"):
                session.start_profile(**profile_kwargs)
                started = True

            # 2) run chat (do NOT pass profile args into chat)
            return session.chat(
                image_paths=imgs,
                prompts=pr,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        finally:
            # 3) stop profile (server-side) only if we started successfully
            if started and hasattr(session, "stop_profile"):
                try:
                    session.stop_profile(**profile_kwargs)
                except Exception:
                    # don't hide main exception
                    pass

    # Non-profile path OR non-sglang-http:
    return session.chat(image_paths=imgs, prompts=pr, max_new_tokens=max_new_tokens, **kwargs)


def run_vl_chat(
    model_id: str,
    *,
    image_paths: Union[str, Sequence[str]],
    prompt: Union[str, Sequence[str]],
    backend_name: str = "torch",
    max_new_tokens: int = 128,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    use_amx: bool = False,
    print_model_info: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    base_url: Optional[str] = None,
    api: str = "v1",
    api_key: str = "",
    timeout: float = 600.0,
    image_transport: str = "data-url",
    # NEW:
    profile: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[str]:
    session = load_vl_session(
        model_id,
        backend_name=backend_name,
        device=device,
        dtype=dtype,
        use_amx=use_amx,
        print_model_info=print_model_info,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        base_url=base_url,
        api=api,
        api_key=api_key,
        timeout=timeout,
        image_transport=image_transport,
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
