from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

try:  # AMX/ipex optional
    import intel_extension_for_pytorch as ipex  # type: ignore

    IPEX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    IPEX_AVAILABLE = False


def _pick_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _device_kind(device: str) -> str:
    d = (device or "").lower()
    if d.startswith("cuda"):
        return "cuda"
    if d.startswith("cpu"):
        return "cpu"
    return d


def _resolve_target_dtype(dtype: Optional[str], device: str) -> Optional[torch.dtype]:
    # Mirror embedding.py behavior, but keep practical defaults for VL models.
    if not dtype or str(dtype).lower() in {"auto", "none"}:
        if _device_kind(device) == "cuda":
            return torch.float16
        # CPU: prefer bf16 if available, else fp32
        return torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32

    dt = str(dtype).lower()
    if dt in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dt in {"fp16", "float16", "half"}:
        return torch.float16
    if dt in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype} (expected one of: auto|fp16|bf16|fp32)")


@dataclass
class TorchVLRuntime:
    processor: Any
    model: Any


class TorchVisionLanguageClient:
    """Offline vision-language inference via Transformers.

    This is intended for models like `Qwen/Qwen2.5-VL-7B-Instruct`.
    """

    name = "torch"

    def load(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        use_amx: bool = False,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        **kwargs: Any,
    ) -> TorchVLRuntime:
        device_str = _pick_device(device)
        target_dtype = _resolve_target_dtype(dtype, device_str)

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        model_kwargs: Dict[str, Any] = {}
        if target_dtype is not None:
            model_kwargs["torch_dtype"] = target_dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=trust_remote_code, **model_kwargs)

        # Match embedding.py: apply dtype conversion and optional IPEX/AMX optimization.
        kind = _device_kind(device_str)
        if kind == "cpu":
            if use_amx:
                if IPEX_AVAILABLE:
                    model = ipex.optimize(model, dtype=target_dtype, inplace=True)  # type: ignore[name-defined]
                if target_dtype is not None:
                    model = model.to(target_dtype)
        elif kind == "cuda" and target_dtype is not None:
            model = model.to(target_dtype)

        model.to(device_str)
        model.eval()
        return TorchVLRuntime(processor=processor, model=model)

    @torch.inference_mode()
    def chat(
        self,
        runtime: TorchVLRuntime,
        *,
        image_paths: Union[str, Sequence[str]],
        prompts: Union[str, Sequence[str]],
        max_new_tokens: int = 128,
        **generate_kwargs: Any,
    ) -> List[str]:
        if isinstance(image_paths, str):
            image_paths_list = [image_paths]
        else:
            image_paths_list = list(image_paths)

        if isinstance(prompts, str):
            prompts_list = [prompts] * len(image_paths_list)
        else:
            prompts_list = list(prompts)
            if len(prompts_list) != len(image_paths_list):
                raise ValueError("prompts must be a single string or the same length as image_paths")

        images = [Image.open(p).convert("RGB") for p in image_paths_list]

        # Qwen2.5-VL and many VLMs require special placeholder tokens in the text
        # to align image tokens with extracted image features.
        proc = runtime.processor
        if hasattr(proc, "apply_chat_template"):
            texts: List[str] = []
            for img, prompt in zip(images, prompts_list):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                t = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(t)

            inputs = proc(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(runtime.model.device)
        else:
            # Fallback for generic VLMs where processor handles image+text directly.
            inputs = proc(
                images=images,
                text=prompts_list,
                return_tensors="pt",
                padding=True,
            ).to(runtime.model.device)

        generated_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        # Decode only the newly generated tokens (avoid echoing the prompt).
        try:
            input_ids = inputs.get("input_ids")
            if isinstance(input_ids, torch.Tensor) and isinstance(generated_ids, torch.Tensor):
                prompt_len = int(input_ids.shape[1])
                if generated_ids.ndim == 2 and generated_ids.shape[1] >= prompt_len:
                    generated_ids = generated_ids[:, prompt_len:]
        except Exception:
            pass

        outputs = proc.batch_decode(generated_ids, skip_special_tokens=True)
        return [o if isinstance(o, str) else str(o) for o in outputs]
