from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSettings:
    # OpenAI-compatible endpoint (vLLM/sglang/OpenAI/etc)
    openai_base_url: str
    openai_api_key: str
    model: str

    # Agent loop
    max_tool_steps: int


def load_settings() -> AgentSettings:
    base = (os.getenv("AGENT_OPENAI_BASE_URL") or "").strip().rstrip("/")
    if not base:
        base = "http://127.0.0.1:8000/v1"

    key = (os.getenv("AGENT_OPENAI_API_KEY") or "").strip()
    model = (os.getenv("AGENT_MODEL") or "").strip() or ""

    max_steps_raw = (os.getenv("AGENT_MAX_TOOL_STEPS") or "").strip() or "5"
    try:
        max_steps = int(max_steps_raw)
    except Exception:
        max_steps = 5

    return AgentSettings(
        openai_base_url=base,
        openai_api_key=key,
        model=model,
        max_tool_steps=max(0, min(20, max_steps)),
    )
