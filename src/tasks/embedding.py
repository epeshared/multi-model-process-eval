from __future__ import annotations

from typing import Any, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

try:  # AMX/ipex optional
    import intel_extension_for_pytorch as ipex  # type: ignore

    IPEX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    IPEX_AVAILABLE = False

from .embedding_backends.sglang_http import SGLangHTTPEmbeddingClient
from .embedding_backends.sglang_offline import SGLangOfflineEmbeddingClient
from .embedding_backends.vllm_http import VLLMHTTPEmbeddingClient
from .embedding_backends.vllm_offline import VLLMOfflineEmbeddingClient


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
    """Embedding runner with torch, sglang HTTP, and sglang offline backends.

    Supports text (default) and image modalities. Image embeddings are currently
    available only for SGLang HTTP and SGLang offline backends.
    """

    backend = (backend_name or "torch").lower()

    mode = (modality or "text").lower()
    data = inputs if inputs is not None else texts
    if data is None:
        raise ValueError("No inputs provided for embedding")

    if mode != "image":
        text_inputs = [str(t) for t in data]
    else:
        text_inputs = []

    if backend == "torch":
        if mode != "text":
            raise ValueError("Torch backend only supports text embeddings")
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

    elif backend == "sglang":
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
        client = SGLangHTTPEmbeddingClient(
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

    elif backend in {"sglang-offline", "sglang_offline"}:
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
        client = SGLangOfflineEmbeddingClient(
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
            **{k: v for k, v in kwargs.items() if k not in {
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
            }},
        )
        if mode == "image":
            return client.encode_images(data, batch_size=batch_size, normalize=normalize)
        return client.encode(text_inputs, batch_size=batch_size, normalize=normalize)

    elif backend in {"vllm-http", "vllm_openai", "vllm-http-openai"}:
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
        client = VLLMHTTPEmbeddingClient(
            base_url=base_url,
            model=model_id,
            api_key=api_key,
            timeout=timeout,
            encoding_format=encoding_format,
        )
        return client.encode(text_inputs, batch_size=batch_size)

    elif backend in {"vllm", "vllm-offline"}:
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
        client = VLLMOfflineEmbeddingClient(
            model=model_id,
            dtype=kwargs.get("dtype", "auto"),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", kwargs.get("tp_size", 1)),
            device=device or kwargs.get("device", "cuda"),
            max_model_len=kwargs.get("max_model_len", 8192),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.90),
        )
        return client.encode(text_inputs, batch_size=batch_size)

    raise ValueError(f"Unsupported embedding backend: {backend_name}")


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


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-9)
    return summed / denom


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(x.norm(p=2, dim=dim, keepdim=True), min=eps)
