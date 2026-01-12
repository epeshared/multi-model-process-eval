# Qwen3 task file, similar to omni.py
from .qwen3_backends.sglang_http import Qwen3SGLangHTTP
from .qwen3_backends.vllm_http import Qwen3VLLMHTTP
from .qwen3_backends.vllm_offline import Qwen3VLLMOffline

class Qwen3Task:
    def __init__(self, model_name, backend="vllm-http"):
        self.model_name = model_name
        if backend == "sglang-http":
            self.backend = Qwen3SGLangHTTP(model_name)
        elif backend == "vllm-http":
            self.backend = Qwen3VLLMHTTP(model_name)
        elif backend == "vllm-offline":
            self.backend = Qwen3VLLMOffline(model_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def run(self, prompt):
        return self.backend.generate(prompt)
