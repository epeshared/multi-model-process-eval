from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import time

try:
    import requests  # type: ignore

    _REQUESTS_OK = True
except Exception:
    _REQUESTS_OK = False


def _norm_base_url(base_url: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        raise ValueError("base_url is required")
    return b


def _maybe_add_v1(base_url: str) -> str:
    b = _norm_base_url(base_url)
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def _join_url(base: str, path: str) -> str:
    b = (base or "").rstrip("/")
    p = (path or "").lstrip("/")
    return f"{b}/{p}" if p else b


def _extract_chat_text(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message")
        if isinstance(msg, dict) and "content" in msg:
            content = msg.get("content")
            if isinstance(content, str):
                return content
            return str(content)
    if isinstance(resp.get("text"), str):
        return str(resp["text"])
    return str(resp)


class Qwen3SGLangHTTPClient:
    """OpenAI-compatible text chat client for SGLang servers."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 600.0,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        backoff: float = 0.75,
    ) -> None:
        if not _REQUESTS_OK:
            raise RuntimeError("requests is required for the sglang HTTP Qwen3 backend")

        self.base_url_root = _norm_base_url(base_url)
        self.base_url_v1 = _maybe_add_v1(base_url)
        self.model = model
        self.api_key = api_key or ""
        self.timeout = float(timeout)
        self.extra_headers = dict(extra_headers or {})
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = _join_url(self.base_url_v1, "/chat/completions")
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
                try:
                    resp.raise_for_status()
                except Exception as e:
                    status = getattr(resp, "status_code", None)
                    body = ""
                    try:
                        body = resp.text or ""
                    except Exception:
                        body = ""
                    body = body.strip()
                    if len(body) > 2000:
                        body = body[:2000] + "...<truncated>"
                    raise RuntimeError(f"HTTP {status} for {url} (model={self.model}): {body or '<no body>'}") from e

                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected response type: {type(data)}")
                return data
            except Exception as e:  # pragma: no cover - network
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"SGLang HTTP chat failed after {self.max_retries} attempts: {last_err}")

    def chat(
        self,
        *,
        prompts: Sequence[str],
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> List[str]:
        outs: List[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": str(prompt)}]
            payload: Dict[str, Any] = {
                "model": self.model or "default",
                "messages": messages,
                "max_tokens": int(max_new_tokens),
            }
            if temperature is not None:
                payload["temperature"] = float(temperature)
            if top_p is not None:
                payload["top_p"] = float(top_p)
            payload.update(kwargs)
            data = self._post_chat(payload)
            outs.append(_extract_chat_text(data))
        return outs
