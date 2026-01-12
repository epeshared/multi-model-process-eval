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


def _v1_chat_url(base_url: str) -> str:
    b = _norm_base_url(base_url)
    if b.endswith("/v1"):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


def _extract_chat_text(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message")
        if isinstance(msg, dict) and "content" in msg:
            c = msg.get("content")
            return c if isinstance(c, str) else str(c)
    if isinstance(resp.get("text"), str):
        return str(resp["text"])
    return str(resp)


class _NonRetryableHTTPError(RuntimeError):
    pass


class Qwen3VLLMHTTPClient:
    """OpenAI-compatible text chat client for vLLM OpenAI servers."""

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
            raise RuntimeError("requests is required for the vLLM HTTP Qwen3 backend")
        self.base_url = _norm_base_url(base_url)
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
        url = _v1_chat_url(self.base_url)
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
                if resp.status_code >= 400:
                    err_msg = resp.text
                    try:
                        j = resp.json()
                        if isinstance(j, dict):
                            e = j.get("error")
                            if isinstance(e, dict) and e.get("message"):
                                err_msg = str(e.get("message"))
                    except Exception:
                        pass

                    if resp.status_code in {400, 401, 403, 404, 409, 422}:
                        raise _NonRetryableHTTPError(f"vLLM HTTP chat failed ({resp.status_code}): {err_msg}")
                    resp.raise_for_status()

                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected response type: {type(data)}")
                return data
            except _NonRetryableHTTPError:
                raise
            except Exception as e:  # pragma: no cover - network
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"vLLM HTTP chat failed after {self.max_retries} attempts: {last_err}")

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
