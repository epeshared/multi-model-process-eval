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


def _maybe_disable_sglang_amx() -> None:
    """Disable SGLang AMX fast-paths (best-effort).

    On some CPU/NUMA setups, SGLang's compiled `sgl_kernel` extension can segfault
    during startup when initializing CPU thread pinning / shared-memory collectives.

    Setting `SGLANG_DISABLE_AMX=1` forces SGLang to treat AMX as unavailable,
    avoiding the problematic initialization path.
    """

    import os

    if os.getenv("SGLANG_DISABLE_AMX", "0").strip() not in {"1", "true", "True", "yes", "on"}:
        return

    try:
        import sglang.srt.utils.common as common  # type: ignore

        def _false() -> bool:
            return False

        common.cpu_has_amx_support = _false  # type: ignore[attr-defined]

        # Also patch the re-export, since some modules import from sglang.srt.utils.
        import sglang.srt.utils as utils  # type: ignore

        utils.cpu_has_amx_support = _false  # type: ignore[attr-defined]
    except Exception:
        return


_maybe_disable_sglang_amx()
