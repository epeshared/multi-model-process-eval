from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import base64
import mimetypes
import os
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
    # Accept either http://host:port or http://host:port/v1
    b = _norm_base_url(base_url)
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "image/jpeg"


def _to_data_url_from_path(path: str) -> str:
    mime = _guess_mime(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _image_to_url_repr(img: Any, image_transport: str) -> str:
    mode = (image_transport or "data-url").lower()

    if isinstance(img, str):
        s = img.strip()
        if s.startswith("data:"):
            return s
        if s.startswith("http://") or s.startswith("https://"):
            return s
        if s.startswith("file://"):
            p = s[len("file://") :]
            if mode == "path/url":
                return s
            if os.path.exists(p) and os.path.isfile(p):
                return _to_data_url_from_path(p)
            return s
        # local file path
        if os.path.exists(s) and os.path.isfile(s):
            if mode == "path/url":
                # Many servers can't access client filesystem; for shared FS setups, file:// may work.
                return "file://" + os.path.abspath(s)
            # data-url/base64 both become a data-url for chat APIs.
            return _to_data_url_from_path(s)
        # unknown string; return as-is
        return s

    if isinstance(img, (bytes, bytearray)):
        b64 = base64.b64encode(bytes(img)).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    # Fallback: try str()
    return str(img)


def _extract_chat_text(resp: Dict[str, Any]) -> str:
    # OpenAI-style: {choices:[{message:{content:...}}]}
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message")
        if isinstance(msg, dict) and "content" in msg:
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Some servers return a list of content blocks; concatenate text parts.
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                        t = item.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                if parts:
                    return "\n".join(parts)
                return str(content)
    # Some servers return {text: ...}
    if isinstance(resp.get("text"), str):
        return str(resp["text"])
    return str(resp)


class SGLangHTTPVLClient:
    """OpenAI-compatible VL chat client for SGLang servers.

    Uses POST {base_url}/v1/chat/completions.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 600.0,
        image_transport: str = "data-url",
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        backoff: float = 0.75,
    ) -> None:
        if not _REQUESTS_OK:
            raise RuntimeError("requests is required for the sglang HTTP VL backend")
        self.base_url = _maybe_add_v1(base_url)
        self.model = model
        self.api_key = api_key or ""
        self.timeout = float(timeout)
        self.image_transport = (image_transport or "data-url").lower()
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
        url = f"{self.base_url}/chat/completions"
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
                try:
                    resp.raise_for_status()
                except Exception as e:
                    # Surface server details; SGLang often returns useful JSON/text error payloads.
                    status = getattr(resp, "status_code", None)
                    body = ""
                    try:
                        body = resp.text or ""
                    except Exception:
                        body = ""
                    body = body.strip()
                    if len(body) > 2000:
                        body = body[:2000] + "...<truncated>"
                    raise RuntimeError(
                        f"HTTP {status} for {url} (model={self.model}): {body or '<no body>'}"
                    ) from e
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
        image_paths: Sequence[Any],
        prompts: Sequence[str],
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> List[str]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

        outs: List[str] = []
        for img, prompt in zip(image_paths, prompts):
            image_url = _image_to_url_repr(img, self.image_transport)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
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
