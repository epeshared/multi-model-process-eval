from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_settings
from .llm_openai import (
    chat_completions,
    extract_assistant_message,
    normalize_tool_calls,
    parse_tool_arguments,
)
from .skills.bootstrap import register_all
from .skills.registry import invoke_skill, list_skill_specs


app = FastAPI(title="multi-model-process-eval agent service", version="0.1")


class SkillInvokeRequest(BaseModel):
    args: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    enable_tools: bool = True


@app.on_event("startup")
def _startup() -> None:
    register_all()


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/v1/skills")
def v1_skills() -> Dict[str, Any]:
    return {"skills": list_skill_specs()}


@app.post("/v1/skills/{skill_name}")
def v1_invoke_skill(skill_name: str, req: SkillInvokeRequest) -> Dict[str, Any]:
    try:
        out = invoke_skill(skill_name, req.args)
        return {"ok": True, "result": out}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _skills_as_openai_tools() -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for s in list_skill_specs():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description") or "",
                    "parameters": s.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


@app.post("/v1/agent/chat")
def v1_agent_chat(req: ChatRequest) -> Dict[str, Any]:
    settings = load_settings()

    messages: List[Dict[str, Any]] = list(req.messages)
    tools = _skills_as_openai_tools() if req.enable_tools else None

    trace: List[Dict[str, Any]] = []

    for step in range(settings.max_tool_steps if req.enable_tools else 1):
        resp = chat_completions(settings=settings, messages=messages, tools=tools)
        msg = extract_assistant_message(resp)
        messages.append(msg)

        tool_calls = normalize_tool_calls(msg) if req.enable_tools else []
        trace.append({"step": step, "assistant": msg, "tool_calls": tool_calls})

        if not tool_calls:
            break

        for tc in tool_calls:
            name = tc.get("name") or ""
            try:
                args = parse_tool_arguments(tc.get("arguments"))
                result = invoke_skill(name, args)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.get("id") or "",
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            except Exception as e:
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.get("id") or "",
                    "name": name,
                    "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                }
            messages.append(tool_msg)

    return {"ok": True, "messages": messages, "trace": trace}
