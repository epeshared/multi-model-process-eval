from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import dataclasses

try:
    from sglang.srt.server_args import ServerArgs  # type: ignore
    import sglang as sgl  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    raise RuntimeError("sglang is required for the sglang-offline VL backend") from e


class SGLangOfflineVLClient:
    """Local sglang Engine-based VL backend.

    Note: sglang's offline generation APIs can vary across versions. This client
    uses runtime feature-detection to call an available generation method.
    """

    def __init__(
        self,
        model: str,
        *,
        dtype: str = "auto",
        device: str = "cuda",
        tp_size: int = 1,
        dp_size: int = 1,
        random_seed: int = 0,
        trust_remote_code: bool = False,
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        attention_backend: Optional[str] = None,
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
            is_embedding=False,
            enable_torch_compile=enable_torch_compile,
            torch_compile_max_bs=torch_compile_max_bs,
            attention_backend=attention_backend,
            log_level="error",
            **engine_extra_kwargs,
        )

        engine_kwargs = dataclasses.asdict(server_args)
        self.engine = sgl.Engine(**engine_kwargs)  # type: ignore[attr-defined]

    def _extract_texts(self, ret: Any) -> List[str]:
        if ret is None:
            return []
        if isinstance(ret, str):
            return [ret]
        if isinstance(ret, list):
            if not ret:
                return []
            if all(isinstance(x, str) for x in ret):
                return list(ret)
            # list of dicts
            out: List[str] = []
            for x in ret:
                if isinstance(x, dict):
                    for k in ("text", "output", "completion", "content"):
                        if isinstance(x.get(k), str):
                            out.append(x[k])
                            break
                else:
                    out.append(str(x))
            return out
        if isinstance(ret, dict):
            for k in ("text", "output", "completion", "content"):
                if isinstance(ret.get(k), str):
                    return [ret[k]]
        return [str(ret)]

    def chat(
        self,
        *,
        image_paths: Sequence[Any],
        prompts: Sequence[str],
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> List[str]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

        # sglang expects image_data to be List[List[MultimodalDataInputItem]]
        image_data = [[img] for img in image_paths]

        # Try common APIs across sglang versions.
        if hasattr(self.engine, "chat"):
            ret = self.engine.chat(
                prompt=list(prompts),
                image_data=image_data,
                max_new_tokens=int(max_new_tokens),
                **kwargs,
            )
            texts = self._extract_texts(ret)
            return texts if len(texts) == len(prompts) else (texts + [""] * (len(prompts) - len(texts)))

        if hasattr(self.engine, "generate"):
            ret = self.engine.generate(
                prompt=list(prompts),
                image_data=image_data,
                max_new_tokens=int(max_new_tokens),
                **kwargs,
            )
            texts = self._extract_texts(ret)
            return texts if len(texts) == len(prompts) else (texts + [""] * (len(prompts) - len(texts)))

        raise RuntimeError(
            "sglang Engine does not expose a known generation method (chat/generate). "
            "Please use the HTTP backend (backend=sglang) or upgrade sglang."
        )
