from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from PIL import Image
from importlib.metadata import PackageNotFoundError, version


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
        dev = (device or "").lower()
        if dev.startswith("cpu"):
            # Allow CPU only if a CPU build of vLLM is installed.
            try:
                vllm_ver = version("vllm")
            except PackageNotFoundError:
                vllm_ver = ""
            if "cpu" not in (vllm_ver or "").lower():
                raise RuntimeError(
                    "backend=vllm (offline) on CPU requires a CPU build of vLLM. "
                    "Fix: use BACKEND=torch for CPU inference, or install a CPU-capable vLLM build, "
                    "or run a CUDA vLLM server and use BACKEND=vllm-http."
                )

        # Compatibility shim: some vLLM versions import OpenAI SDK types by old names.
        # Newer openai releases expose ChatCompletionToolParam but not ChatCompletionFunctionToolParam.
        try:  # pragma: no cover - best-effort import workaround
            import openai.types.chat as _chat

            if not hasattr(_chat, "ChatCompletionFunctionToolParam") and hasattr(_chat, "ChatCompletionToolParam"):
                setattr(_chat, "ChatCompletionFunctionToolParam", getattr(_chat, "ChatCompletionToolParam"))
        except Exception:
            pass
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            msg = str(e)
            hint = ""
            if "openai.types.chat" in msg or "ChatCompletionFunctionToolParam" in msg:
                hint = (
                    " (Hint: your 'openai' package version is likely incompatible with the installed vLLM. "
                    "Try upgrading 'openai' to a newer 1.x release, or install a vLLM version that matches your openai.)"
                )
            raise RuntimeError(f"vllm not installed or import failed: {e}{hint}")

        self._SamplingParams = SamplingParams
        self._tokenizer: Any = None
        try:  # pragma: no cover - optional helper
            from transformers import AutoTokenizer  # type: ignore

            # Use the same model path/id so chat template tokens match.
            self._tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception:
            self._tokenizer = None

        init_kwargs: Dict[str, Any] = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            enforce_eager=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if max_model_len is not None and max_model_len > 0:
            init_kwargs["max_model_len"] = int(max_model_len)
        init_kwargs.update(llm_kwargs)

        # vLLM kwargs are not stable across versions; best-effort retries.
        # - Some versions don't accept certain knobs (especially CPU builds).
        try_kwargs = dict(init_kwargs)

        def _try_init(kwargs: Dict[str, Any]) -> Any:
            return LLM(**kwargs)

        try:
            self.llm = _try_init(try_kwargs)
        except TypeError:
            for k in ("gpu_memory_utilization", "enforce_eager"):
                if k in try_kwargs:
                    try_kwargs.pop(k, None)
                    try:
                        self.llm = _try_init(try_kwargs)
                        break
                    except TypeError:
                        continue
            else:
                # Some vLLM versions require VLLM_DEVICE env var rather than a constructor kwarg.
                import os

                if device:
                    os.environ.setdefault("VLLM_DEVICE", device)
                self.llm = _try_init(try_kwargs)

    def _format_multimodal_prompt(self, prompt: str) -> str:
        """Return a prompt string containing required multimodal placeholders.

        For Qwen2.5-VL, vLLM expects image placeholder tokens (e.g.
        <|vision_start|><|image_pad|><|vision_end|>) to appear in the prompt so it
        can align `multi_modal_data['image']` with the text.
        """

        p = prompt or ""

        # If user already provided a full template, don't re-wrap.
        if any(t in p for t in ("<|vision_start|>", "<|image_pad|>", "<|im_start|>")):
            return p

        tok = self._tokenizer
        if tok is not None and hasattr(tok, "apply_chat_template"):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": p},
                        ],
                    }
                ]
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        # Fallback: prepend the common Qwen-VL placeholder.
        return "<|vision_start|><|image_pad|><|vision_end|>" + p

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
            reqs.append({"prompt": self._format_multimodal_prompt(prompt), "multi_modal_data": {"image": pil_img}})

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
