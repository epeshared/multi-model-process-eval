from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
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


def _v1_chat_url(base_url: str) -> str:
    b = _norm_base_url(base_url)
    if b.endswith("/v1"):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


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
        if os.path.exists(s) and os.path.isfile(s):
            if mode == "path/url":
                return "file://" + os.path.abspath(s)
            return _to_data_url_from_path(s)
        return s

    if isinstance(img, (bytes, bytearray)):
        b64 = base64.b64encode(bytes(img)).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    return str(img)


def _extract_chat_text(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message")
        if isinstance(msg, dict) and "content" in msg:
            c = msg.get("content")
            return c if isinstance(c, str) else str(c)
    return str(resp)


class VLLMHTTPVLClient:
    """OpenAI-compatible VL chat client for vLLM OpenAI servers."""

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
            raise RuntimeError("requests is required for the vLLM HTTP VL backend")
        self.base_url = _norm_base_url(base_url)
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
        url = _v1_chat_url(self.base_url)
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected response type: {type(data)}")
                return data
            except Exception as e:  # pragma: no cover - network
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"vLLM HTTP chat failed after {self.max_retries} attempts: {last_err}")

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
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
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
