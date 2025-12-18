from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union
import dataclasses

import torch

try:
    from sglang.srt.server_args import ServerArgs  # type: ignore
    import sglang as sgl  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    raise RuntimeError("sglang is required for the sglang-offline embedding backend") from e


class SGLangOfflineEmbeddingClient:
    @staticmethod
    def _extract_embeddings(ret: Any) -> List[List[float]]:
        if ret is None:
            return []

        if hasattr(ret, "embeddings"):
            embs = getattr(ret, "embeddings")
            if isinstance(embs, list):
                return [list(map(float, e)) for e in embs]

        if isinstance(ret, dict):
            if "embedding" in ret:
                return [list(map(float, ret["embedding"]))]
            if "embeddings" in ret:
                embs = ret["embeddings"]
                if isinstance(embs, list):
                    return [list(map(float, e)) for e in embs]
            if "data" in ret and isinstance(ret["data"], list):
                rows = sorted(ret["data"], key=lambda d: (d or {}).get("index", 0))
                out: List[List[float]] = []
                for r in rows:
                    if not isinstance(r, dict) or "embedding" not in r:
                        raise RuntimeError(f"Unexpected item in Engine.encode data: {type(r)}")
                    out.append(list(map(float, r["embedding"])))
                return out

        if isinstance(ret, list):
            if not ret:
                return []
            if isinstance(ret[0], dict) and "embedding" in ret[0]:
                return [list(map(float, (r or {})["embedding"])) for r in ret]
            if isinstance(ret[0], (list, tuple)):
                return [list(map(float, r)) for r in ret]

        raise RuntimeError(
            "Unexpected return from sglang.Engine.encode. "
            f"type={type(ret)} value_preview={str(ret)[:2000]}"
        )

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        device: str = "cuda",
        tp_size: int = 1,
        dp_size: int = 1,
        random_seed: int = 0,
        trust_remote_code: bool = False,
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        attention_backend: Optional[str] = None,
        is_embedding: bool = True,
        enable_torch_compile: bool = True,
        torch_compile_max_bs: int = 32,
        **engine_extra_kwargs: Dict[str, Any],
    ) -> None:
        server_args = ServerArgs(
            model_path=model,
            dtype=dtype,
            device=device,
            tp_size=tp_size,
            dp_size=dp_size,
            random_seed=random_seed,
            trust_remote_code=trust_remote_code,
            quantization=quantization,
            revision=revision,
            is_embedding=is_embedding,
            enable_torch_compile=enable_torch_compile,
            torch_compile_max_bs=torch_compile_max_bs,
            attention_backend=attention_backend,
            log_level="error",
            **engine_extra_kwargs,
        )

        engine_kwargs = dataclasses.asdict(server_args)
        self.engine = sgl.Engine(**engine_kwargs)  # type: ignore[attr-defined]

    @torch.inference_mode()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 128,
        normalize: bool = True,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)

        all_chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            ret = self.engine.encode(batch)
            vecs = self._extract_embeddings(ret)
            if len(vecs) != len(batch):
                raise RuntimeError(
                    f"sglang.Engine.encode returned {len(vecs)} embeddings for {len(batch)} inputs. "
                    "This usually indicates a backend/model mismatch or an incompatible sglang version."
                )
            embs = torch.tensor(vecs, dtype=torch.float32)
            if normalize:
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_chunks.append(embs)

        return torch.cat(all_chunks, dim=0)

    @torch.inference_mode()
    def encode_images(
        self,
        images: List[Union[Any, str, dict]],
        batch_size: int = 128,
        normalize: bool = True,
    ) -> torch.Tensor:
        if not images:
            return torch.empty(0, 0)

        all_chunks: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            dummy_prompts = [""] * len(batch_images)
            ret = self.engine.encode(dummy_prompts, image_data=batch_images)
            vecs = self._extract_embeddings(ret)
            if len(vecs) != len(batch_images):
                raise RuntimeError(
                    f"sglang.Engine.encode returned {len(vecs)} embeddings for {len(batch_images)} images. "
                    "This usually indicates the model does not support image embeddings in offline mode."
                )

            embs = torch.tensor(vecs, dtype=torch.float32)
            if normalize:
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_chunks.append(embs)

        return torch.cat(all_chunks, dim=0)
