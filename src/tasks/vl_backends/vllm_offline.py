from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from PIL import Image


class VLLMOfflineVLClient:
    """Local vLLM vision-language backend.

    This is best-effort because vLLM multimodal APIs can vary by version.
    """

    def __init__(
        self,
        model: str,
        *,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        device: str = "cuda",
        max_model_len: Optional[int] = 8192,
        gpu_memory_utilization: float = 0.90,
        **llm_kwargs: Dict[str, Any],
    ) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(f"vllm not installed or import failed: {e}")

        self._SamplingParams = SamplingParams

        init_kwargs: Dict[str, Any] = dict(
            model=model,
            task="generate",
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            enforce_eager=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if max_model_len is not None and max_model_len > 0:
            init_kwargs["max_model_len"] = int(max_model_len)
        init_kwargs.update(llm_kwargs)

        try:
            self.llm = LLM(**init_kwargs)
        except TypeError:
            # Older vLLM versions may require VLLM_DEVICE env var.
            import os

            os.environ["VLLM_DEVICE"] = device
            self.llm = LLM(**init_kwargs)

    def _extract_texts(self, outputs: Any) -> List[str]:
        out: List[str] = []
        if outputs is None:
            return out
        if isinstance(outputs, list):
            for o in outputs:
                try:
                    # vLLM RequestOutput: o.outputs[0].text
                    if hasattr(o, "outputs") and o.outputs:
                        out.append(str(o.outputs[0].text))
                    else:
                        out.append(str(o))
                except Exception:
                    out.append(str(o))
            return out
        return [str(outputs)]

    def chat(
        self,
        *,
        image_paths: Sequence[Any],
        prompts: Sequence[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> List[str]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

        sampling_params = self._SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            **kwargs,
        )

        # vLLM multimodal is version-dependent. Try the most common schema:
        # llm.generate([{"prompt": ..., "multi_modal_data": {"image": PIL.Image}}], sampling_params=...)
        reqs: List[Dict[str, Any]] = []
        for img, prompt in zip(image_paths, prompts):
            if isinstance(img, Image.Image):
                pil_img = img
            elif isinstance(img, str):
                pil_img = Image.open(img).convert("RGB")
            else:
                # try best-effort conversion
                pil_img = Image.open(str(img)).convert("RGB")
            reqs.append({"prompt": prompt, "multi_modal_data": {"image": pil_img}})

        try:
            llm_any: Any = self.llm
            outputs = llm_any.generate(reqs, sampling_params=sampling_params)
            texts = self._extract_texts(outputs)
            return texts if len(texts) == len(prompts) else (texts + [""] * (len(prompts) - len(texts)))
        except TypeError as e:
            raise RuntimeError(
                "Your vLLM version does not appear to support multimodal offline generation via the "
                "{prompt, multi_modal_data} input schema. Consider using the HTTP backend (backend=vllm-http) "
                "or upgrade vLLM. Original error: " + str(e)
            )
