"""Process-wide startup hooks.

Python automatically imports `sitecustomize` (if present on sys.path) during
startup, including for multiprocessing-spawned child processes.

We use this to apply a small compatibility shim between vLLM and the OpenAI
Python SDK type definitions.
"""

from __future__ import annotations


def _patch_openai_types() -> None:
    try:
        import openai.types.chat as chat  # type: ignore

        # vLLM versions may import a symbol that was renamed/removed in newer
        # OpenAI SDK versions.
        if not hasattr(chat, "ChatCompletionFunctionToolParam") and hasattr(chat, "ChatCompletionToolParam"):
            setattr(chat, "ChatCompletionFunctionToolParam", getattr(chat, "ChatCompletionToolParam"))
    except Exception:
        # Best-effort: never block startup.
        return


_patch_openai_types()
