from __future__ import annotations

from typing import Any, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from .embedding_backends.sglang_http import SGLangHTTPEmbeddingClient
from .embedding_backends.sglang_offline import SGLangOfflineEmbeddingClient
from .embedding_backends.vllm_http import VLLMHTTPEmbeddingClient
from .embedding_backends.vllm_offline import VLLMOfflineEmbeddingClient


@torch.inference_mode()
def run_embedding(
    model_id: str,
    texts: List[str],
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
    **kwargs: Any,
) -> torch.Tensor:
    """Embedding runner with torch, sglang HTTP, and sglang offline backends."""

    backend = (backend_name or "torch").lower()

    if backend == "torch":
        return _run_torch_embedding(
            model_id=model_id,
            texts=texts,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )

    if backend == "sglang":
        if not base_url:
            raise ValueError("base_url is required for the sglang HTTP embedding backend")
        client = SGLangHTTPEmbeddingClient(
            base_url=base_url,
            model=model_id,
            api=api,
            api_key=api_key,
            timeout=timeout,
            image_transport=image_transport,
        )
        return client.encode(texts, batch_size=batch_size, normalize=normalize)

    if backend in {"sglang-offline", "sglang_offline"}:
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
                "random_seed",
                "trust_remote_code",
                "quantization",
                "revision",
                "attention_backend",
                "enable_torch_compile",
                "torch_compile_max_bs",
            }},
        )
        return client.encode(texts, batch_size=batch_size, normalize=normalize)

    if backend in {"vllm-http", "vllm_openai", "vllm-http-openai"}:
        if not base_url:
            raise ValueError("base_url is required for the vLLM HTTP embedding backend")
        client = VLLMHTTPEmbeddingClient(
            base_url=base_url,
            model=model_id,
            api_key=api_key,
            timeout=timeout,
            encoding_format=encoding_format,
        )
        return client.encode(texts, batch_size=batch_size)

    if backend in {"vllm", "vllm-offline"}:
        client = VLLMOfflineEmbeddingClient(
            model=model_id,
            dtype=kwargs.get("dtype", "auto"),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", kwargs.get("tp_size", 1)),
            device=device or kwargs.get("device", "cuda"),
            max_model_len=kwargs.get("max_model_len", 8192),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.90),
        )
        return client.encode(texts, batch_size=batch_size)

    raise ValueError(f"Unsupported embedding backend: {backend_name}")


def _run_torch_embedding(
    model_id: str,
    texts: List[str],
    device: Optional[str],
    normalize: bool,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
) -> torch.Tensor:
    if not texts:
        return torch.empty(0, 0)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()
    target_device = torch.device(device) if device else model.device
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
