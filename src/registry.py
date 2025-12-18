from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .tasks.audio import run_audio_classification, run_speech_recognition
from .tasks.diffusion import run_general_diffusion, run_stable_diffusion_v1, run_stable_diffusion_xl
from .tasks.embedding import run_embedding
from .tasks.multimodal import run_vision_language_chat
from .tasks.text_generation import run_text_generation
from .tasks.translation import run_translation
from .tasks.text_classification import run_text_classification
from .tasks.video import run_video_caption
from .tasks.vision import (
    run_clip_similarity,
    run_image_classification,
    run_image_text_matching,
    run_image_to_text,
    run_owlvit_detection,
)


@dataclass
class ModelSpec:
    key: str
    model_id: str
    task: str
    backends: List[str]
    runner: Callable[..., Any]
    description: str


def _tg(backends: List[str]) -> List[str]:
    return backends


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "qwen3-embedding-4b": ModelSpec(
        key="qwen3-embedding-4b",
        model_id="Qwen/Qwen3-Embedding-4B",
        task="embedding",
        backends=["torch", "sglang", "sglang-offline", "vllm", "vllm-http"],
        runner=run_embedding,
        description="Text embedding model (large).",
    ),
    "qwen3-embedding-0.6b": ModelSpec(
        key="qwen3-embedding-0.6b",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        task="embedding",
        backends=["torch", "sglang", "sglang-offline", "vllm", "vllm-http"],
        runner=run_embedding,
        description="Text embedding model (small).",
    ),
    "qwen3-1.7b": ModelSpec(
        key="qwen3-1.7b",
        model_id="Qwen/Qwen3-1.7B",
        task="text-generation",
        backends=["torch", "vllm", "sglang"],
        runner=run_text_generation,
        description="Qwen3 text generation model.",
    ),
    "qwen2.5-omni-7b": ModelSpec(
        key="qwen2.5-omni-7b",
        model_id="Qwen/Qwen2.5-Omni-7B",
        task="text-generation",
        backends=["torch", "vllm", "sglang"],
        runner=run_text_generation,
        description="Qwen2.5 Omni 7B text generation.",
    ),
    "qwen2.5-vl-7b-instruct": ModelSpec(
        key="qwen2.5-vl-7b-instruct",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        task="vision-language",
        backends=["torch"],
        runner=run_vision_language_chat,
        description="Vision-language instruction model.",
    ),
    "openai-clip-vit-base-patch32": ModelSpec(
        key="openai-clip-vit-base-patch32",
        model_id="openai/clip-vit-base-patch32",
        task="clip-similarity",
        backends=["torch"],
        runner=run_clip_similarity,
        description="CLIP image-text similarity.",
    ),
    "nsfw-image-detection": ModelSpec(
        key="nsfw-image-detection",
        model_id="nsfw_image_detection",
        task="image-classification",
        backends=["torch"],
        runner=run_image_classification,
        description="NSFW image classifier.",
    ),
    "aesthetics-predictor-v1": ModelSpec(
        key="aesthetics-predictor-v1",
        model_id="aesthetics-predictor-v1-vit-base-patch16",
        task="image-classification",
        backends=["torch"],
        runner=run_image_classification,
        description="Aesthetics predictor v1.",
    ),
    "blip2-opt-2.7b": ModelSpec(
        key="blip2-opt-2.7b",
        model_id="Salesforce/blip2-opt-2.7b",
        task="image-to-text",
        backends=["torch"],
        runner=run_image_to_text,
        description="BLIP2 image captioning.",
    ),
    "stable-diffusion-v1-4": ModelSpec(
        key="stable-diffusion-v1-4",
        model_id="CompVis/stable-diffusion-v1-4",
        task="image-generation",
        backends=["torch"],
        runner=run_stable_diffusion_v1,
        description="Stable Diffusion v1-4 image generation.",
    ),
    "klue-roberta-intent": ModelSpec(
        key="klue-roberta-intent",
        model_id="bespin-global/klue-roberta-small-3i4k-intent-classification",
        task="text-classification",
        backends=["torch"],
        runner=run_text_classification,
        description="Intent classification via sequence classification pipeline.",
    ),
    "opus-mt-zh-en": ModelSpec(
        key="opus-mt-zh-en",
        model_id="Helsinki-NLP/opus-mt-zh-en",
        task="translation",
        backends=["torch"],
        runner=run_translation,
        description="Chinese to English MT.",
    ),
    "opus-mt-en-zh": ModelSpec(
        key="opus-mt-en-zh",
        model_id="Helsinki-NLP/opus-mt-en-zh",
        task="translation",
        backends=["torch"],
        runner=run_translation,
        description="English to Chinese MT.",
    ),
    "financial-sentiment": ModelSpec(
        key="financial-sentiment",
        model_id="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        task="text-classification",
        backends=["torch"],
        runner=run_text_classification,
        description="Financial sentiment classifier.",
    ),
    "topic-classification": ModelSpec(
        key="topic-classification",
        model_id="dstefa/roberta-base_topic_classification_nyt_news",
        task="text-classification",
        backends=["torch"],
        runner=run_text_classification,
        description="NYT topic classification.",
    ),
    "qwen-audio": ModelSpec(
        key="qwen-audio",
        model_id="Qwen/Qwen-Audio",
        task="speech-recognition",
        backends=["torch"],
        runner=run_speech_recognition,
        description="Qwen audio speech recognition.",
    ),
    "flan-t5-summarization": ModelSpec(
        key="flan-t5-summarization",
        model_id="mrm8488/flan-t5-large-finetuned-openai-summarize_from_feedback",
        task="text-generation",
        backends=["torch", "vllm", "sglang"],
        runner=run_text_generation,
        description="FLAN-T5 summarization.",
    ),
    "stable-diffusion-xl": ModelSpec(
        key="stable-diffusion-xl",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        task="image-generation",
        backends=["torch"],
        runner=run_stable_diffusion_xl,
        description="Stable Diffusion XL base.",
    ),
    "video-blip-ego4d": ModelSpec(
        key="video-blip-ego4d",
        model_id="kpyu/video-blip-opt-2.7b-ego4d",
        task="video-caption",
        backends=["torch"],
        runner=run_video_caption,
        description="Video BLIP captioning (ego4d).",
    ),
    "ast-audioset": ModelSpec(
        key="ast-audioset",
        model_id="MIT/ast-finetuned-audioset-10-10-0.4593",
        task="audio-classification",
        backends=["torch"],
        runner=run_audio_classification,
        description="AudioSet AST classifier.",
    ),
    "aesthetics-predictor-v2": ModelSpec(
        key="aesthetics-predictor-v2",
        model_id="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
        task="image-classification",
        backends=["torch"],
        runner=run_image_classification,
        description="Aesthetics predictor v2.",
    ),
    "watermark-detector": ModelSpec(
        key="watermark-detector",
        model_id="amrul-hzz/watermark_detector",
        task="image-classification",
        backends=["torch"],
        runner=run_image_classification,
        description="Watermark detector classifier.",
    ),
    "owlvit-base": ModelSpec(
        key="owlvit-base",
        model_id="google/owlvit-base-patch32",
        task="object-detection",
        backends=["torch"],
        runner=run_owlvit_detection,
        description="Zero-shot object detector (OwlViT).",
    ),
    "blip-itm-base": ModelSpec(
        key="blip-itm-base",
        model_id="Salesforce/blip-itm-base-coco",
        task="image-text-matching",
        backends=["torch"],
        runner=run_image_text_matching,
        description="BLIP image-text matching.",
    ),
    "llava-vicuna-7b": ModelSpec(
        key="llava-vicuna-7b",
        model_id="llava-hf/llava-v1.6-vicuna-7b-hf",
        task="vision-language",
        backends=["torch"],
        runner=run_vision_language_chat,
        description="LLaVA vision-language model.",
    ),
    "pythia-6.9b": ModelSpec(
        key="pythia-6.9b",
        model_id="EleutherAI/pythia-6.9b-deduped",
        task="text-generation",
        backends=["torch", "vllm", "sglang"],
        runner=run_text_generation,
        description="Pythia text generation.",
    ),
}


def get_spec(model_key: str) -> ModelSpec:
    key = model_key.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Model '{model_key}' is not registered")
    return MODEL_REGISTRY[key]


def run_model(model_key: str, backend: str, **kwargs: Any) -> Any:
    spec = get_spec(model_key)
    if backend not in spec.backends:
        raise ValueError(f"Backend {backend} is not supported for model {spec.key}")

    if spec.task in {"text-generation"}:
        return spec.runner(
            model_id=spec.model_id,
            backend_name=backend,
            prompt=kwargs.get("prompt", kwargs.get("text", "")),
            max_new_tokens=kwargs.get("max_new_tokens", 128),
            temperature=kwargs.get("temperature", 0.7),
            device=kwargs.get("device"),
        )

    if spec.task == "embedding":
        embedding_kwargs = {k: v for k, v in kwargs.items() if k not in {"texts", "text"}}
        return spec.runner(
            model_id=spec.model_id,
            backend_name=backend,
            texts=kwargs.get("texts") or [kwargs.get("text", "")],
            **embedding_kwargs,
        )

    if spec.task == "image-classification":
        return spec.runner(
            model_id=spec.model_id,
            image_path=kwargs["image"],
            device=kwargs.get("device"),
        )

    if spec.task == "clip-similarity":
        return spec.runner(
            model_id=spec.model_id,
            image_path=kwargs["image"],
            text_queries=kwargs.get("texts") or [kwargs.get("text", "")],
            device=kwargs.get("device"),
        )

    if spec.task == "image-to-text":
        return spec.runner(
            model_id=spec.model_id,
            image_path=kwargs["image"],
            device=kwargs.get("device"),
            max_new_tokens=kwargs.get("max_new_tokens", 64),
        )

    if spec.task == "image-generation":
        return spec.runner(
            model_id=spec.model_id,
            prompt=kwargs.get("prompt", ""),
            device=kwargs.get("device"),
            num_inference_steps=kwargs.get("num_inference_steps", 30),
        )

    if spec.task == "translation":
        return spec.runner(
            model_id=spec.model_id,
            text=kwargs.get("text", ""),
            device=kwargs.get("device"),
        )

    if spec.task == "audio-classification":
        return spec.runner(
            model_id=spec.model_id,
            audio_path=kwargs["audio"],
            device=kwargs.get("device"),
        )

    if spec.task == "speech-recognition":
        return spec.runner(
            model_id=spec.model_id,
            audio_path=kwargs["audio"],
            device=kwargs.get("device"),
        )

    if spec.task == "object-detection":
        return spec.runner(
            model_id=spec.model_id,
            image_path=kwargs["image"],
            text_queries=kwargs.get("texts") or [kwargs.get("text", "")],
            device=kwargs.get("device"),
        )

    if spec.task == "image-text-matching":
        return spec.runner(
            model_id=spec.model_id,
            image_path=kwargs["image"],
            text=kwargs.get("text", ""),
            device=kwargs.get("device"),
        )

    if spec.task == "vision-language":
        return spec.runner(
            model_id=spec.model_id,
            image_path=kwargs["image"],
            prompt=kwargs.get("prompt", ""),
            device=kwargs.get("device"),
            max_new_tokens=kwargs.get("max_new_tokens", 128),
        )

    if spec.task == "video-caption":
        return spec.runner(
            model_id=spec.model_id,
            video_path=kwargs["video"],
            device=kwargs.get("device"),
        )

    if spec.task == "text-classification":
        return spec.runner(
            model_id=spec.model_id,
            text=kwargs.get("text", ""),
            device=kwargs.get("device"),
        )

    raise ValueError(f"Unsupported task type: {spec.task}")


def available_models() -> List[ModelSpec]:
    return list(MODEL_REGISTRY.values())
