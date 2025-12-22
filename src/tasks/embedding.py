from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:  # AMX/ipex optional
    import intel_extension_for_pytorch as ipex  # type: ignore

    IPEX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    IPEX_AVAILABLE = False

from .embedding_backends.sglang_http import SGLangHTTPEmbeddingClient
from .embedding_backends.sglang_offline import SGLangOfflineEmbeddingClient
from .embedding_backends.vllm_http import VLLMHTTPEmbeddingClient
from .embedding_backends.vllm_offline import VLLMOfflineEmbeddingClient


def _parse_torch_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if not dtype:
        return None
    dt = dtype.lower()
    if dt in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dt in {"fp16", "float16", "half"}:
        return torch.float16
    if dt in {"fp32", "float32", "float"}:
        return torch.float32
    return None


def _normalize_embedding_inputs(
    *,
    texts: Optional[List[str]],
    inputs: Optional[List[Any]],
    modality: str,
) -> tuple[List[Any], List[str], str]:
    mode = (modality or "text").lower()
    data = inputs if inputs is not None else texts
    if data is None:
        raise ValueError("No inputs provided for embedding")
    if mode != "image":
        text_inputs = [str(t) for t in data]
    else:
        text_inputs = []
    return list(data), text_inputs, mode


@dataclass
class TorchEmbeddingSession:
    model_id: str
    model: Any
    tokenizer: Any | None
    processor: Any | None
    device: torch.device
    target_dtype: Optional[torch.dtype]
    is_clip: bool

    def encode_text(
        self,
        texts: List[str],
        *,
        normalize: bool,
        batch_size: int,
        max_length: int,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)

        if self.is_clip:
            if self.processor is None:
                raise RuntimeError("CLIP processor not initialized")
            enc = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # processor returns tensors and sometimes lists; move tensors
            enc = {k: v.to(self.device) for k, v in enc.items() if hasattr(v, "to")}
            feats = self.model.get_text_features(**enc)
            if normalize:
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            return feats

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        hidden_states = self.model(**encoded).last_hidden_state
        pooled = _mean_pooling(hidden_states, encoded["attention_mask"])
        if normalize:
            pooled = _l2_normalize(pooled, dim=1)
        return pooled

    def encode_images(
        self,
        images: List[Any],
        *,
        normalize: bool,
        batch_size: int,
    ) -> torch.Tensor:
        if not images:
            return torch.empty(0, 0)
        if not self.is_clip:
            raise ValueError("Torch backend only supports image embeddings for CLIP-style models")
        if self.processor is None:
            raise RuntimeError("CLIP processor not initialized")

        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError(f"Pillow (PIL) is required for CLIP image embeddings: {e}")

        chunks: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            pil_images: List[Any] = []
            for x in batch:
                if isinstance(x, str):
                    pil_images.append(Image.open(x).convert("RGB"))
                elif hasattr(x, "convert"):
                    pil_images.append(x.convert("RGB"))
                else:
                    raise ValueError(f"Unsupported image input type for CLIP torch backend: {type(x)}")

            enc = self.processor(images=pil_images, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items() if hasattr(v, "to")}
            if self.target_dtype is not None and "pixel_values" in enc:
                enc["pixel_values"] = enc["pixel_values"].to(dtype=self.target_dtype)
            feats = self.model.get_image_features(**enc)
            if normalize:
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            chunks.append(feats)

        return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 0)


def load_embedding_session(
    model_id: str,
    *,
    backend_name: str = "torch",
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    print_model_info: bool = False,
    base_url: Optional[str] = None,
    api: str = "v1",
    api_key: str = "",
    timeout: float = 120.0,
    image_transport: str = "data-url",
    encoding_format: Optional[str] = None,
    use_amx: bool = False,
    dtype: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Load/init a reusable embedding session.

    Intended for benchmarks where you want to exclude model/client load time.
    Use with embed_with_session(session, ...).
    """

    backend = (backend_name or "torch").lower()

    if backend == "torch":
        is_clip = _is_clip_model(model_id, trust_remote_code=trust_remote_code)
        target_dtype = _parse_torch_dtype(dtype)

        if is_clip:
            try:
                from transformers import CLIPModel, CLIPProcessor
            except Exception as e:
                raise RuntimeError(f"transformers CLIPModel/CLIPProcessor required for CLIP embeddings: {e}")

            processor = CLIPProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            model = CLIPModel.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()
            tokenizer = None
        else:
            processor = None
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()

        target_device = torch.device(device) if device else model.device

        if device == "cpu":
            if use_amx:
                if IPEX_AVAILABLE:
                    model = ipex.optimize(model, dtype=target_dtype, inplace=True)
                model = model.to(target_dtype)
        elif device == "cuda" and target_dtype is not None:
            model = model.to(target_dtype)

        model.to(target_device)

        if print_model_info:
            try:
                n_params = None
                if hasattr(model, "parameters"):
                    n_params = sum(int(p.numel()) for p in model.parameters())
                cfg = getattr(model, "config", None)
                model_type = getattr(cfg, "model_type", None) if cfg is not None else None
                hidden_size = getattr(cfg, "hidden_size", None) if cfg is not None else None
                max_pos = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
                print(
                    "[embedding.load] backend=torch "
                    f"model_id={model_id} model_class={type(model).__name__} "
                    f"device={getattr(model, 'device', None)} dtype={getattr(model, 'dtype', None)} "
                    f"clip={is_clip} model_type={model_type} hidden_size={hidden_size} max_pos={max_pos} "
                    f"n_params={n_params}"
                )
            except Exception as e:
                print(f"[embedding.load] print_model_info failed: {e}")

        return (
            "torch",
            TorchEmbeddingSession(
                model_id=model_id,
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                device=target_device,
                target_dtype=target_dtype,
                is_clip=is_clip,
            ),
        )

    if backend == "sglang":
        if not base_url:
            raise ValueError("base_url is required for the sglang HTTP embedding backend")
        if print_model_info:
            print(f"[embedding.load] backend=sglang-http base_url={base_url} model_id={model_id} api={api}")
        return SGLangHTTPEmbeddingClient(
            base_url=base_url,
            model=model_id,
            api=api,
            api_key=api_key,
            timeout=timeout,
            image_transport=image_transport,
        )

    if backend in {"sglang-offline", "sglang_offline"}:
        # Keep backward-compat: offline backends historically took dtype from kwargs.
        if dtype is not None and "dtype" not in kwargs:
            kwargs["dtype"] = dtype
        if print_model_info:
            print(
                f"[embedding.load] backend=sglang-offline model_id={model_id} "
                f"device={device or kwargs.get('device', 'cuda')} dtype={kwargs.get('dtype', 'auto')} "
                f"tp_size={kwargs.get('tp_size', kwargs.get('tensor_parallel_size', 1))}"
            )
        return SGLangOfflineEmbeddingClient(
            model=model_id,
            dtype=kwargs.get("dtype", "auto"),
            device=device or kwargs.get("device", "cuda"),
            tp_size=kwargs.get("tp_size", kwargs.get("tensor_parallel_size", 1)),
            dp_size=kwargs.get("dp_size", 1),
            random_seed=kwargs.get("random_seed", 0),
            trust_remote_code=kwargs.get("trust_remote_code", False),
            quantization=kwargs.get("quantization"),
            revision=kwargs.get("revision"),
            attention_backend=kwargs.get("attention_backend"),
            is_embedding=True,
            enable_torch_compile=kwargs.get("enable_torch_compile", True),
            torch_compile_max_bs=kwargs.get("torch_compile_max_bs", 32),
            **_filtered_sglang_offline_kwargs(kwargs),
        )

    if backend in {"vllm-http", "vllm_openai", "vllm-http-openai"}:
        if not base_url:
            raise ValueError("base_url is required for the vLLM HTTP embedding backend")
        if print_model_info:
            print(f"[embedding.load] backend=vllm-http base_url={base_url} model_id={model_id} encoding_format={encoding_format}")
        return VLLMHTTPEmbeddingClient(
            base_url=base_url,
            model=model_id,
            api_key=api_key,
            timeout=timeout,
            encoding_format=encoding_format,
        )

    if backend in {"vllm", "vllm-offline"}:
        if dtype is not None and "dtype" not in kwargs:
            kwargs["dtype"] = dtype
        if print_model_info:
            print(
                f"[embedding.load] backend=vllm-offline model_id={model_id} "
                f"device={device or kwargs.get('device', 'cuda')} dtype={kwargs.get('dtype', 'auto')} "
                f"tp_size={kwargs.get('tensor_parallel_size', kwargs.get('tp_size', 1))} "
                f"max_model_len={kwargs.get('max_model_len', 8192)} gpu_mem_util={kwargs.get('gpu_memory_utilization', 0.90)}"
            )
        return VLLMOfflineEmbeddingClient(
            model=model_id,
            dtype=kwargs.get("dtype", "auto"),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", kwargs.get("tp_size", 1)),
            device=device or kwargs.get("device", "cuda"),
            max_model_len=kwargs.get("max_model_len", 8192),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.90),
        )

    raise ValueError(f"Unsupported embedding backend: {backend_name}")


@torch.inference_mode()
def embed_with_session(
    session: Any,
    *,
    texts: Optional[List[str]] = None,
    inputs: Optional[List[Any]] = None,
    modality: str = "text",
    normalize: bool = True,
    batch_size: int = 128,
    max_length: int = 512,
    **kwargs: Any,
) -> torch.Tensor:
    data, text_inputs, mode = _normalize_embedding_inputs(texts=texts, inputs=inputs, modality=modality)

    if isinstance(session, tuple) and len(session) == 2 and session[0] == "torch":
        torch_sess: TorchEmbeddingSession = session[1]
        _log_backend_call(
            "torch(session)",
            model_id=torch_sess.model_id,
            device=str(torch_sess.device),
            normalize=normalize,
            batch_size=batch_size,
            max_length=max_length,
            dtype=str(torch_sess.target_dtype) if torch_sess.target_dtype is not None else None,
            clip=torch_sess.is_clip,
            modality=mode,
            inputs=data,
        )
        chunks: List[torch.Tensor] = []
        if mode == "image":
            for i in range(0, len(data), batch_size):
                feats = torch_sess.encode_images(
                    list(data[i : i + batch_size]),
                    normalize=normalize,
                    batch_size=batch_size,
                )
                chunks.append(feats.cpu())
        else:
            for i in range(0, len(text_inputs), batch_size):
                feats = torch_sess.encode_text(
                    list(text_inputs[i : i + batch_size]),
                    normalize=normalize,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                chunks.append(feats.cpu())
        hidden = chunks[0].shape[-1] if chunks else 0
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0, hidden)

    # Non-torch backends: expect client has encode()/encode_images().
    if mode == "image":
        if not hasattr(session, "encode_images"):
            raise ValueError("This embedding backend does not support image embeddings")
        return session.encode_images(data, batch_size=batch_size, normalize=normalize, **kwargs)

    if not hasattr(session, "encode"):
        raise ValueError("This embedding backend does not support text embeddings")
    return session.encode(text_inputs, batch_size=batch_size, normalize=normalize, **kwargs)


def _log_backend_call(name: str, **params: Any) -> None:
    safe: dict = {}
    for k, v in params.items():
        if k in {"texts", "inputs", "text_inputs", "data"}:
            try:
                safe[k] = f"<list len={len(v) if v is not None else 0}>"
            except Exception:
                safe[k] = "<list>"
        elif k in {"api_key"}:
            safe[k] = "<hidden>" if v else ""
        else:
            safe[k] = v
    print(f"[run_embedding] backend={name} params={safe}")


def _is_clip_model(model_id: str, trust_remote_code: bool = True) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        model_type = getattr(cfg, "model_type", "")
        if isinstance(model_type, str) and model_type.lower() in {"clip", "open_clip"}:
            return True
        arch = getattr(cfg, "architectures", None)
        if isinstance(arch, (list, tuple)) and any("CLIP" in str(a) for a in arch):
            return True
    except Exception:
        pass

    s = (model_id or "").lower()
    return "clip" in s


@torch.inference_mode()
def run_embedding(
    model_id: str,
    texts: Optional[List[str]] = None,
    inputs: Optional[List[Any]] = None,
    modality: str = "text",
    backend_name: str = "torch",
    device: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 128,
    max_length: int = 512,
    trust_remote_code: bool = True,
    base_url: Optional[str] = None,
    api: str = "v1",
    api_key: str = "",
    timeout: float = 120.0,
    image_transport: str = "data-url",
    encoding_format: Optional[str] = None,
    use_amx: bool = False,
    dtype: Optional[str] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Embedding runner across torch / sglang / vllm.

    Modalities:
    - text: supported by all backends
    - image: supported by sglang (http/offline) and torch when using CLIP-style models
    """

    backend = (backend_name or "torch").lower()
    mode = (modality or "text").lower()

    data = inputs if inputs is not None else texts
    if data is None:
        raise ValueError("No inputs provided for embedding")

    text_inputs: List[str]
    if mode != "image":
        text_inputs = [str(t) for t in data]
    else:
        text_inputs = []

    if backend == "torch":
        return _run_backend_torch(
            model_id=model_id,
            data=data,
            text_inputs=text_inputs,
            mode=mode,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            use_amx=use_amx,
            dtype=dtype,
        )

    if backend == "sglang":
        return _run_backend_sglang_http(
            model_id=model_id,
            data=data,
            text_inputs=text_inputs,
            mode=mode,
            base_url=base_url,
            api=api,
            api_key=api_key,
            timeout=timeout,
            image_transport=image_transport,
            batch_size=batch_size,
            normalize=normalize,
        )

    if backend in {"sglang-offline", "sglang_offline"}:
        return _run_backend_sglang_offline(
            model_id=model_id,
            data=data,
            text_inputs=text_inputs,
            mode=mode,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            kwargs=kwargs,
        )

    if backend in {"vllm-http", "vllm_openai", "vllm-http-openai"}:
        return _run_backend_vllm_http(
            model_id=model_id,
            data=data,
            text_inputs=text_inputs,
            mode=mode,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            encoding_format=encoding_format,
            batch_size=batch_size,
        )

    if backend in {"vllm", "vllm-offline"}:
        return _run_backend_vllm_offline(
            model_id=model_id,
            data=data,
            text_inputs=text_inputs,
            mode=mode,
            device=device,
            batch_size=batch_size,
            kwargs=kwargs,
        )

    raise ValueError(f"Unsupported embedding backend: {backend_name}")


def _run_backend_torch(
    *,
    model_id: str,
    data: List[Any],
    text_inputs: List[str],
    mode: str,
    device: Optional[str],
    normalize: bool,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    use_amx: bool,
    dtype: Optional[str],
) -> torch.Tensor:
    is_clip = _is_clip_model(model_id, trust_remote_code=trust_remote_code)
    if mode != "text" and not is_clip:
        raise ValueError("Torch backend only supports text embeddings (unless using a CLIP-style model)")
    _log_backend_call(
        "torch",
        model_id=model_id,
        device=device,
        normalize=normalize,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        texts=text_inputs,
        use_amx=use_amx,
        dtype=dtype,
        clip=is_clip,
        modality=mode,
    )
    if is_clip:
        if mode == "image":
            return _run_torch_clip_image_embedding(
                model_id=model_id,
                images=list(data),
                device=device,
                normalize=normalize,
                batch_size=batch_size,
                trust_remote_code=trust_remote_code,
                use_amx=use_amx,
                dtype=dtype,
            )
        return _run_torch_clip_text_embedding(
            model_id=model_id,
            texts=text_inputs,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            use_amx=use_amx,
            dtype=dtype,
        )

    return _run_torch_embedding(
        model_id=model_id,
        texts=text_inputs,
        device=device,
        normalize=normalize,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        use_amx=use_amx,
        dtype=dtype,
    )


def _run_backend_sglang_http(
    *,
    model_id: str,
    data: List[Any],
    text_inputs: List[str],
    mode: str,
    base_url: Optional[str],
    api: str,
    api_key: str,
    timeout: float,
    image_transport: str,
    batch_size: int,
    normalize: bool,
) -> torch.Tensor:
    if not base_url:
        raise ValueError("base_url is required for the sglang HTTP embedding backend")
    _log_backend_call(
        "sglang-http",
        model_id=model_id,
        base_url=base_url,
        api=api,
        timeout=timeout,
        image_transport=image_transport,
        batch_size=batch_size,
        normalize=normalize,
        mode=mode,
        inputs=data,
    )
    client: SGLangHTTPEmbeddingClient = SGLangHTTPEmbeddingClient(
        base_url=base_url,
        model=model_id,
        api=api,
        api_key=api_key,
        timeout=timeout,
        image_transport=image_transport,
    )
    if mode == "image":
        return client.encode_images(data, batch_size=batch_size, normalize=normalize)
    return client.encode(text_inputs, batch_size=batch_size, normalize=normalize)


def _filtered_sglang_offline_kwargs(kwargs: dict) -> dict:
    forbidden = {
        "dtype",
        "device",
        "tp_size",
        "dp_size",
        "tensor_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
        "random_seed",
        "trust_remote_code",
        "quantization",
        "revision",
        "attention_backend",
        "enable_torch_compile",
        "torch_compile_max_bs",
    }
    return {k: v for k, v in kwargs.items() if k not in forbidden}


def _run_backend_sglang_offline(
    *,
    model_id: str,
    data: List[Any],
    text_inputs: List[str],
    mode: str,
    device: Optional[str],
    batch_size: int,
    normalize: bool,
    kwargs: dict,
) -> torch.Tensor:
    _log_backend_call(
        "sglang-offline",
        model_id=model_id,
        device=device or kwargs.get("device", "cuda"),
        dtype=kwargs.get("dtype", "auto"),
        tp_size=kwargs.get("tp_size", 1),
        dp_size=kwargs.get("dp_size", 1),
        random_seed=kwargs.get("random_seed", 0),
        quantization=kwargs.get("quantization"),
        revision=kwargs.get("revision"),
        attention_backend=kwargs.get("attention_backend"),
        enable_torch_compile=kwargs.get("enable_torch_compile", True),
        torch_compile_max_bs=kwargs.get("torch_compile_max_bs", 32),
        batch_size=batch_size,
        normalize=normalize,
        mode=mode,
        inputs=data,
    )
    client: SGLangOfflineEmbeddingClient = SGLangOfflineEmbeddingClient(
        model=model_id,
        dtype=kwargs.get("dtype", "auto"),
        device=device or kwargs.get("device", "cuda"),
        tp_size=kwargs.get("tp_size", 1),
        dp_size=kwargs.get("dp_size", 1),
        random_seed=kwargs.get("random_seed", 0),
        trust_remote_code=kwargs.get("trust_remote_code", False),
        quantization=kwargs.get("quantization"),
        revision=kwargs.get("revision"),
        attention_backend=kwargs.get("attention_backend"),
        is_embedding=True,
        enable_torch_compile=kwargs.get("enable_torch_compile", True),
        torch_compile_max_bs=kwargs.get("torch_compile_max_bs", 32),
        **_filtered_sglang_offline_kwargs(kwargs),
    )
    if mode == "image":
        return client.encode_images(data, batch_size=batch_size, normalize=normalize)
    return client.encode(text_inputs, batch_size=batch_size, normalize=normalize)


def _run_backend_vllm_http(
    *,
    model_id: str,
    data: List[Any],
    text_inputs: List[str],
    mode: str,
    base_url: Optional[str],
    api_key: str,
    timeout: float,
    encoding_format: Optional[str],
    batch_size: int,
) -> torch.Tensor:
    if mode == "image":
        raise ValueError("vLLM HTTP backend does not support image embeddings")
    if not base_url:
        raise ValueError("base_url is required for the vLLM HTTP embedding backend")
    _log_backend_call(
        "vllm-http",
        model_id=model_id,
        base_url=base_url,
        timeout=timeout,
        encoding_format=encoding_format,
        batch_size=batch_size,
        mode=mode,
        inputs=data,
    )
    client: VLLMHTTPEmbeddingClient = VLLMHTTPEmbeddingClient(
        base_url=base_url,
        model=model_id,
        api_key=api_key,
        timeout=timeout,
        encoding_format=encoding_format,
    )
    return client.encode(text_inputs, batch_size=batch_size)


def _run_backend_vllm_offline(
    *,
    model_id: str,
    data: List[Any],
    text_inputs: List[str],
    mode: str,
    device: Optional[str],
    batch_size: int,
    kwargs: dict,
) -> torch.Tensor:
    if mode == "image":
        raise ValueError("vLLM offline backend does not support image embeddings")
    _log_backend_call(
        "vllm-offline",
        model_id=model_id,
        device=device or kwargs.get("device", "cuda"),
        dtype=kwargs.get("dtype", "auto"),
        tensor_parallel_size=kwargs.get("tensor_parallel_size", kwargs.get("tp_size", 1)),
        max_model_len=kwargs.get("max_model_len", 8192),
        gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.90),
        batch_size=batch_size,
        mode=mode,
        inputs=data,
    )
    client: VLLMOfflineEmbeddingClient = VLLMOfflineEmbeddingClient(
        model=model_id,
        dtype=kwargs.get("dtype", "auto"),
        tensor_parallel_size=kwargs.get("tensor_parallel_size", kwargs.get("tp_size", 1)),
        device=device or kwargs.get("device", "cuda"),
        max_model_len=kwargs.get("max_model_len", 8192),
        gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.90),
    )
    return client.encode(text_inputs, batch_size=batch_size)


def _run_torch_embedding(
    model_id: str,
    texts: List[str],
    device: Optional[str],
    normalize: bool,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    use_amx: bool,
    dtype: Optional[str],
) -> torch.Tensor:
    if not texts:
        return torch.empty(0, 0)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()

    target_device = torch.device(device) if device else model.device

    target_dtype: Optional[torch.dtype] = None
    if dtype:
        dt = dtype.lower()
        if dt in {"bf16", "bfloat16"}:
            target_dtype = torch.bfloat16
        elif dt in {"fp16", "float16", "half"}:
            target_dtype = torch.float16
        elif dt in {"fp32", "float32", "float"}:
            target_dtype = torch.float32

    if device == "cpu":
        if use_amx:
            if IPEX_AVAILABLE:
                model = ipex.optimize(model, dtype=target_dtype, inplace=True)
            model = model.to(target_dtype)
    elif device == "cuda" and target_dtype is not None:
        model = model.to(target_dtype)

    model.to(target_device)

    chunks: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(target_device)
        hidden_states = model(**encoded).last_hidden_state
        pooled = _mean_pooling(hidden_states, encoded["attention_mask"])
        if normalize:
            pooled = _l2_normalize(pooled, dim=1)
        chunks.append(pooled.cpu())

    hidden = getattr(model.config, "hidden_size", 0)
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, hidden)


def _run_torch_clip_text_embedding(
    model_id: str,
    texts: List[str],
    device: Optional[str],
    normalize: bool,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    use_amx: bool,
    dtype: Optional[str],
) -> torch.Tensor:
    if not texts:
        return torch.empty(0, 0)

    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        raise RuntimeError(f"transformers CLIPModel/CLIPProcessor required for CLIP embeddings: {e}")

    processor = CLIPProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = CLIPModel.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()

    target_device = torch.device(device) if device else model.device

    target_dtype: Optional[torch.dtype] = None
    if dtype:
        dt = dtype.lower()
        if dt in {"bf16", "bfloat16"}:
            target_dtype = torch.bfloat16
        elif dt in {"fp16", "float16", "half"}:
            target_dtype = torch.float16
        elif dt in {"fp32", "float32", "float"}:
            target_dtype = torch.float32

    if device == "cpu":
        if use_amx:
            if IPEX_AVAILABLE:
                model = ipex.optimize(model, dtype=target_dtype, inplace=True)
            model = model.to(target_dtype)
    elif device == "cuda" and target_dtype is not None:
        model = model.to(target_dtype)

    model.to(target_device)

    chunks: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = processor(
            text=batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(target_device) for k, v in enc.items() if hasattr(v, "to")}
        feats = model.get_text_features(**enc)
        if normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        chunks.append(feats.cpu())

    hidden = chunks[0].shape[-1] if chunks else 0
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, hidden)


def _run_torch_clip_image_embedding(
    model_id: str,
    images: List[Any],
    device: Optional[str],
    normalize: bool,
    batch_size: int,
    trust_remote_code: bool,
    use_amx: bool,
    dtype: Optional[str],
) -> torch.Tensor:
    if not images:
        return torch.empty(0, 0)

    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"Pillow (PIL) is required for CLIP image embeddings: {e}")

    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        raise RuntimeError(f"transformers CLIPModel/CLIPProcessor required for CLIP embeddings: {e}")

    processor = CLIPProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = CLIPModel.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()

    target_device = torch.device(device) if device else model.device

    target_dtype: Optional[torch.dtype] = None
    if dtype:
        dt = dtype.lower()
        if dt in {"bf16", "bfloat16"}:
            target_dtype = torch.bfloat16
        elif dt in {"fp16", "float16", "half"}:
            target_dtype = torch.float16
        elif dt in {"fp32", "float32", "float"}:
            target_dtype = torch.float32

    if device == "cpu":
        if use_amx:
            if IPEX_AVAILABLE:
                model = ipex.optimize(model, dtype=target_dtype, inplace=True)
            model = model.to(target_dtype)
    elif device == "cuda" and target_dtype is not None:
        model = model.to(target_dtype)

    model.to(target_device)

    chunks: List[torch.Tensor] = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pil_images: List[Any] = []
        for x in batch:
            if isinstance(x, str):
                pil_images.append(Image.open(x).convert("RGB"))
            elif hasattr(x, "convert"):
                pil_images.append(x.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image input type for CLIP torch backend: {type(x)}")

        enc = processor(images=pil_images, return_tensors="pt")
        enc = {k: v.to(target_device) for k, v in enc.items() if hasattr(v, "to")}

        # Cast pixel_values to target dtype if requested.
        if target_dtype is not None and "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(dtype=target_dtype)

        feats = model.get_image_features(**enc)
        if normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        chunks.append(feats.cpu())

    hidden = chunks[0].shape[-1] if chunks else 0
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, hidden)


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-9)
    return summed / denom


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(x.norm(p=2, dim=dim, keepdim=True), min=eps)
