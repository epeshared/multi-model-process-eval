# multi-model-process-eval

提供一套统一的代码骨架，用 sglang、vllm、torch 三类后端跑不同模态的模型。所有模型都挂在注册表里，通过各自的脚本入口运行。文本向量现已支持 torch 本地、sglang HTTP/离线 Engine，以及 vLLM 本地与 OpenAI 兼容 HTTP。

## 目录

- src/backends: 三个后端封装（torch/vllm/sglang）。
- src/tasks: 按任务的最小推理逻辑（生成、分类、翻译、扩散等）。
- src/registry.py: 模型注册表与统一分发。
- src/cli.py: 通用 CLI 入口。
- scripts/: 分任务的脚本入口。

## 支持模型与对应脚本

- 文本生成: qwen3-1.7b, qwen2.5-omni-7b, flan-t5-summarization, pythia-6.9b → scripts/run_text_generation.py
- 文本向量: qwen3-embedding-4b, qwen3-embedding-0.6b → scripts/run_embedding.py
- 文本分类: klue-roberta-intent, financial-sentiment, topic-classification → scripts/run_text_classification.py
- 翻译: opus-mt-zh-en, opus-mt-en-zh → scripts/run_translation.py
- 视觉: openai-clip-vit-base-patch32, nsfw-image-detection, aesthetics-predictor-v1, aesthetics-predictor-v2, watermark-detector, blip2-opt-2.7b, owlvit-base, blip-itm-base → scripts/run_vision.py
- 视觉语言: qwen2.5-vl-7b-instruct, llava-vicuna-7b → scripts/run_multimodal.py
- 扩散: stable-diffusion-v1-4, stable-diffusion-xl → scripts/run_diffusion.py
- 音频: qwen-audio, ast-audioset → scripts/run_audio.py
- 视频: video-blip-ego4d → scripts/run_video.py

## 依赖

```bash
pip install -r requirements.txt
```

按需安装 vllm/sglang（仅文本生成模型需要）。

## 示例

文本生成（torch 默认）:

```bash
python scripts/run_text_generation.py --model qwen3-1.7b --prompt "Hello" --max-new-tokens 64
```

文本生成（vllm）:

```bash
python scripts/run_text_generation.py --model qwen3-1.7b --backend vllm --prompt "你好" --max-new-tokens 64
```

向量（torch，本地模型）:

```bash
python scripts/run_embedding.py --model qwen3-embedding-4b --backend torch --texts "今天天气不错" "How are you"
```

向量（sglang HTTP /v1/embeddings，多模态也可）:

```bash
python scripts/run_embedding.py --model qwen3-embedding-4b --backend sglang \
  --base-url http://127.0.0.1:30000 --api v1 --texts "hello" "world"
```

向量（sglang 离线 Engine，本地模型路径）:

```bash
python scripts/run_embedding.py --model /path/to/model --backend sglang-offline \
  --device cuda:0 --batch-size 64 --texts "hello" "world"
```

向量（vLLM OpenAI 兼容 HTTP /v1/embeddings）:

```bash
python scripts/run_embedding.py --model Qwen/Qwen3-Embedding-0.6B --backend vllm-http \
  --base-url http://127.0.0.1:8000/v1 --texts "hello" "world"
```

向量（vLLM 本地 embed）:

```bash
python scripts/run_embedding.py --model Qwen/Qwen3-Embedding-0.6B --backend vllm \
  --tp-size 1 --max-model-len 8192 --texts "hello" "world"
```

图像 NSFW:

```bash
python scripts/run_vision.py --model nsfw-image-detection --image path/to/img.jpg
```

CLIP 相似度:

```bash
python scripts/run_vision.py --model openai-clip-vit-base-patch32 --image img.jpg --texts "a cat" "a dog"
```

扩散生成:

```bash
python scripts/run_diffusion.py --model stable-diffusion-v1-4 --prompt "a sunset city"
```

音频分类:

```bash
python scripts/run_audio.py --model ast-audioset --audio sample.wav
```

多模态问答:

```bash
python scripts/run_multimodal.py --model llava-vicuna-7b --image demo.jpg --prompt "Describe the scene"
```

统一入口（可用作底层调用）:

```bash
python -m src.cli --model-key qwen3-1.7b --backend torch --prompt "Hello"
```
