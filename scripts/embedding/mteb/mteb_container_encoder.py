from __future__ import annotations

from dataclasses import dataclass
import time
import math
from typing import Any, Iterable, Optional

import numpy as np
import torch

from src.tasks.embedding import embed_with_session, load_embedding_session


def _apply_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text
    if prefix.endswith(" ") or prefix.endswith("\n"):
        return f"{prefix}{text}"
    return f"{prefix} {text}"


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return list(x)


def _mteb_model_meta_name(backend: str, model_id: str) -> str:
    """Return a name that satisfies MTEB's 'org/model_name' constraint."""
    model_id = (model_id or "").strip()
    if "/" in model_id:
        return model_id
    org = (backend or "container").replace("_", "-").replace(":", "-")
    return f"{org}/{model_id or 'unknown-model'}"


@dataclass
class ContainerMTEBConfig:
    backend: str
    base_url: str
    model_id: str
    api: str = "v1"  # sglang only
    api_key: str = ""
    timeout: float = 120.0
    encoding_format: str | None = None  # vllm-http only
    normalize: bool = True
    batch_size: int = 128
    max_length: int = 512
    query_prefix: str = ""
    document_prefix: str = ""
    profile: bool = False
    profile_kwargs: dict[str, Any] | None = None


class ContainerMTEBEncoder:
    """MTEB encoder wrapper that routes encoding to existing container backends.

    This is intentionally lightweight and reuses:
    - `src.tasks.embedding.load_embedding_session`
    - `src.tasks.embedding.embed_with_session`

    Note: This module requires `mteb` at runtime, but does not import it at
    module import time to keep it as an optional dependency.
    """

    def __init__(self, cfg: ContainerMTEBConfig):
        self.cfg = cfg
        self.model_name = _mteb_model_meta_name(cfg.backend, cfg.model_id)
        self.revision = "container"

        # Runtime stats (filled during encode)
        self.total_texts_encoded: int = 0
        self.total_batches: int = 0
        self.total_encode_time_s: float = 0.0

        self._session = load_embedding_session(
            cfg.model_id,
            backend_name=cfg.backend,
            base_url=cfg.base_url,
            api=cfg.api,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            encoding_format=cfg.encoding_format,
        )

        # Provide MTEB model metadata if MTEB looks for it.
        # Import lazily so projects without mteb don't fail to import this file.
        from mteb.models.model_meta import ModelMeta

        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=self.model_name,
            revision=self.revision,
            reference=None,
            release_date=None,
            languages=None,
            license=None,
            framework=["API"],
            training_datasets=None,
            similarity_fn_name=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            use_instructions=None,
            modalities=["text"],
        )

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode MTEB DataLoader[BatchedInput] -> np.ndarray."""

        # MTEB uses PromptType.query / PromptType.document
        prompt_value = getattr(prompt_type, "value", None)
        if prompt_value == "query":
            prefix = self.cfg.query_prefix
        elif prompt_value in {"document", "passage"}:
            prefix = self.cfg.document_prefix
        else:
            prefix = ""

        # Flatten DataLoader batches -> list[str]
        texts: list[str] = []
        for batch in inputs:
            if not isinstance(batch, dict) or "text" not in batch:
                raise ValueError(
                    f"Unsupported MTEB batch format. Expected dict with 'text'. Got: {type(batch)} keys={getattr(batch, 'keys', lambda: None)()}"
                )
            batch_texts = _as_list(batch["text"])
            texts.extend([_apply_prefix(str(t), prefix) for t in batch_texts])

        start = time.perf_counter()
        embs = embed_with_session(
            self._session,
            texts=texts,
            modality="text",
            normalize=self.cfg.normalize,
            batch_size=self.cfg.batch_size,
            max_length=self.cfg.max_length,
            profile=self.cfg.profile,
            profile_kwargs=self.cfg.profile_kwargs,
        )
        elapsed = time.perf_counter() - start

        self.total_encode_time_s += float(elapsed)
        self.total_texts_encoded += len(texts)
        if self.cfg.batch_size and self.cfg.batch_size > 0:
            self.total_batches += int(math.ceil(len(texts) / float(self.cfg.batch_size)))

        if isinstance(embs, torch.Tensor):
            return embs.detach().cpu().float().numpy()
        return np.asarray(embs, dtype=np.float32)

    def get_encoding_stats(self) -> dict[str, Any]:
        tps = None
        if self.total_encode_time_s > 0:
            tps = self.total_texts_encoded / self.total_encode_time_s
        return {
            "total_texts_encoded": int(self.total_texts_encoded),
            "total_batches": int(self.total_batches),
            "batch_size": int(self.cfg.batch_size),
            "encode_time_s": float(self.total_encode_time_s),
            "tps_texts_per_s": float(tps) if tps is not None else None,
            "normalize": bool(self.cfg.normalize),
        }

    def similarity(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        a = np.asarray(embeddings1, dtype=np.float32)
        b = np.asarray(embeddings2, dtype=np.float32)
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return a @ b.T

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        a = np.asarray(embeddings1, dtype=np.float32)
        b = np.asarray(embeddings2, dtype=np.float32)
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return np.sum(a * b, axis=1)
