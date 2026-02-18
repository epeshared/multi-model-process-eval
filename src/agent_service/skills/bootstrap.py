from __future__ import annotations

from .registry import SkillSpec, register
from . import scale_run, scale_status, remote_preflight, log_analyze


def register_all() -> None:
    register(
        SkillSpec(
            name="scale_run_fix_token_len",
            description="Run/resume the embedding scale-test (fix_token_len) via existing runner.",
            parameters_schema=scale_run.SPEC,
            handler=scale_run.handler,
        )
    )
    register(
        SkillSpec(
            name="scale_status_fix_token_len",
            description="Summarize a scale_id status by inspecting <result_root>/<scale_id>/.",
            parameters_schema=scale_status.SPEC,
            handler=scale_status.handler,
        )
    )
    register(
        SkillSpec(
            name="remote_preflight_fix_token_len",
            description="SSH preflight checks for hosts in the scale-test config (conda/repo/result dirs).",
            parameters_schema=remote_preflight.SPEC,
            handler=remote_preflight.handler,
        )
    )
    register(
        SkillSpec(
            name="log_analyze",
            description="Rule-based log analyzer for common failure patterns (ssh timeout/conda missing/etc).",
            parameters_schema=log_analyze.SPEC,
            handler=log_analyze.handler,
        )
    )
