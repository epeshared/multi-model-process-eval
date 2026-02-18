from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from .config import AgentSettings


def _auth_headers(settings: AgentSettings) -> Dict[str, str]:
    if settings.openai_api_key:
        return {"Authorization": f"Bearer {settings.openai_api_key}"}
    return {}


def chat_completions(
    *,
    settings: AgentSettings,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> Dict[str, Any]:
    if not settings.model:
        raise ValueError("AGENT_MODEL is required for /v1/agent/chat")

    url = f"{settings.openai_base_url}/chat/completions"
    payload: Dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, headers={"Content-Type": "application/json", **_auth_headers(settings)}, json=payload)
        r.raise_for_status()
        return r.json()


def extract_assistant_message(resp: Dict[str, Any]) -> Dict[str, Any]:
    choices = resp.get("choices") or []
    if not choices:
        return {"role": "assistant", "content": ""}
    msg = (choices[0] or {}).get("message") or {}
    return msg


def normalize_tool_calls(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    # OpenAI-style: message.tool_calls = [{id,type,function:{name,arguments}}]
    tcs = msg.get("tool_calls") or []
    out: List[Dict[str, Any]] = []
    for tc in tcs:
        fn = (tc or {}).get("function") or {}
        out.append(
            {
                "id": (tc or {}).get("id") or "",
                "name": str(fn.get("name") or ""),
                "arguments": fn.get("arguments") or "{}",
            }
        )
    # Older function_call: message.function_call
    if not out and msg.get("function_call"):
        fn = msg.get("function_call") or {}
        out.append({"id": "", "name": str(fn.get("name") or ""), "arguments": fn.get("arguments") or "{}"})
    return out


def parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        s = arguments.strip() or "{}"
        return json.loads(s)
    raise ValueError("unsupported tool arguments")
