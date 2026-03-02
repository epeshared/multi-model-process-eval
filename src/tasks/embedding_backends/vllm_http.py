from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import time
import base64
import ipaddress
import mimetypes
import os
import sys
from array import array
from urllib.parse import urlparse

import torch

try:
    import requests  # type: ignore
    _REQUESTS_OK = True
except Exception:
    _REQUESTS_OK = False


def _is_local_base_url(base_url: str) -> bool:
    try:
        p = urlparse(base_url)
    except Exception:
        return False
    host = (p.hostname or "").strip().lower()
    if not host:
        return False
    if host in {"localhost"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_unspecified)


def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, p=2, dim=1)


class VLLMHTTPEmbeddingClient:
    """OpenAI-compatible HTTP embeddings client for vLLM servers."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 600.0,
        encoding_format: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        backoff: float = 0.75,
        image_transport: str = "data-url",
    ) -> None:
        if not _REQUESTS_OK:
            raise RuntimeError("requests is required for the vLLM HTTP embedding backend")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.encoding_format = encoding_format
        self.extra_headers = dict(extra_headers or {})
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)

        self.image_transport = (image_transport or "data-url").lower()

        self.session = requests.Session()
        if _is_local_base_url(self.base_url):
            # Avoid proxying localhost traffic via http_proxy/https_proxy.
            self.session.trust_env = False

        # Tag for upper layers if needed.
        self._backend_tag = "vllm-http"

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h

    @staticmethod
    def _decode_base64_embedding(s: str) -> List[float]:
        # OpenAI-compatible: base64-encoded little-endian float32 array.
        raw = base64.b64decode(s)
        arr = array("f")
        arr.frombytes(raw)
        if sys.byteorder != "little":
            arr.byteswap()
        return [float(x) for x in arr]

    @classmethod
    def _extract_embeddings(cls, data: Dict[str, Any]) -> List[List[float]]:
        rows = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        out: List[List[float]] = []
        for r in rows:
            emb = r.get("embedding")
            if isinstance(emb, list):
                out.append([float(x) for x in emb])
            elif isinstance(emb, str):
                out.append(cls._decode_base64_embedding(emb))
            else:
                raise RuntimeError(f"Unexpected embedding type in response: {type(emb)}")
        return out

    def _post_embeddings(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        urls = [f"{self.base_url}/v1/embeddings", f"{self.base_url}/embeddings"]
        base_payload: Dict[str, Any] = {"model": self.model, "input": inputs}

        # If encoding_format isn't explicitly set, try base64 first.
        # This avoids JSON serialization failures on the server when embeddings contain NaNs.
        payload_variants: List[Dict[str, Any]] = []
        if self.encoding_format is not None:
            p = dict(base_payload)
            if self.encoding_format:
                p["encoding_format"] = self.encoding_format
            payload_variants.append(p)
        else:
            p_base64 = dict(base_payload)
            p_base64["encoding_format"] = "base64"
            payload_variants.append(p_base64)
            payload_variants.append(dict(base_payload))

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                for url in urls:
                    for payload in payload_variants:
                        resp = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
                        if resp.status_code == 404:
                            break

                        # If server doesn't understand base64 encoding_format, fall back to default.
                        if (
                            self.encoding_format is None
                            and payload.get("encoding_format") == "base64"
                            and resp.status_code in (400, 422)
                        ):
                            continue

                        resp.raise_for_status()
                        data = resp.json()
                        return self._extract_embeddings(data)
                raise RuntimeError("vLLM embeddings endpoint not found (tried /v1/embeddings and /embeddings)")
            except Exception as e:  # pragma: no cover - network
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"vLLM HTTP embeddings failed after {self.max_retries} attempts: {last_err}")

    def _post_embeddings_multimodal(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[List[float]]:
        """DEPRECATED: kept for compatibility.

        vLLM multimodal embeddings should be sent using the EmbeddingChatRequest
        shape (messages + image_url). See `_post_embeddings_chat`.
        """

        raise RuntimeError(
            "vLLM HTTP multimodal embeddings payload format changed: expected EmbeddingChatRequest (messages + image_url)."
        )

    def _post_embeddings_chat(self, messages: List[Dict[str, Any]]) -> List[List[float]]:
        """Embeddings via EmbeddingChatRequest.

        vLLM (>=0.16) exposes /v1/embeddings but supports multimodal inputs via
        chat-style `messages` (with content parts like {type: image_url}).
        """

        urls = [f"{self.base_url}/v1/embeddings", f"{self.base_url}/embeddings"]
        base_payload: Dict[str, Any] = {"model": self.model, "messages": messages}

        payload_variants: List[Dict[str, Any]] = []
        if self.encoding_format is not None:
            p = dict(base_payload)
            if self.encoding_format:
                p["encoding_format"] = self.encoding_format
            payload_variants.append(p)
        else:
            p_base64 = dict(base_payload)
            p_base64["encoding_format"] = "base64"
            payload_variants.append(p_base64)
            payload_variants.append(dict(base_payload))

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                for url in urls:
                    for payload in payload_variants:
                        resp = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
                        if resp.status_code == 404:
                            break

                        if resp.status_code >= 400:
                            # Include body text for easier debugging (schemas, proxy HTML, etc.).
                            msg = (resp.text or "").strip()
                            msg = msg[:2000] + ("..." if len(msg) > 2000 else "")
                            raise RuntimeError(f"vLLM /v1/embeddings error {resp.status_code}: {msg}")

                        data = resp.json()
                        return self._extract_embeddings(data)
                raise RuntimeError("vLLM embeddings endpoint not found (tried /v1/embeddings and /embeddings)")
            except Exception as e:  # pragma: no cover - network
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"vLLM HTTP embeddings(chat) failed after {self.max_retries} attempts: {last_err}")

    def _image_to_image_url(self, img: Any) -> Any:
        rep = self._image_to_repr(img)

        # vLLM expects `image_url.url` to be a URL-like string. Data URLs work.
        if isinstance(rep, str):
            s = rep.strip()
            if s.startswith("data:") or s.startswith("http://") or s.startswith("https://"):
                return rep

            # If user asked for base64 transport, wrap it as a data URL.
            if self.image_transport == "base64" and s:
                return f"data:application/octet-stream;base64,{s}"

        return rep

    def _image_to_repr(self, img: Any) -> Any:
        if self.image_transport == "path/url":
            return img

        if self.image_transport == "base64":
            if isinstance(img, str):
                s = img.strip()
                if s.startswith("data:"):
                    comma = s.find(",")
                    return s[comma + 1 :] if comma >= 0 else s
                if s.startswith("http://") or s.startswith("https://"):
                    return s
                if os.path.exists(s) and os.path.isfile(s):
                    with open(s, "rb") as f:
                        return base64.b64encode(f.read()).decode("ascii")
                return s
            if isinstance(img, (bytes, bytearray)):
                return base64.b64encode(bytes(img)).decode("ascii")
            return img

        # data-url
        if isinstance(img, str):
            s = img.strip()
            if s.startswith("data:") or s.startswith("http://") or s.startswith("https://"):
                return s
            if os.path.exists(s) and os.path.isfile(s):
                mime, _ = mimetypes.guess_type(s)
                if not mime:
                    mime = "application/octet-stream"
                with open(s, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                return f"data:{mime};base64,{b64}"
            return s

        if isinstance(img, (bytes, bytearray)):
            b64 = base64.b64encode(bytes(img)).decode("ascii")
            return f"data:application/octet-stream;base64,{b64}"
        return img

    @torch.inference_mode()
    def encode_images(self, images: List[Any], batch_size: int = 32, normalize: bool = True) -> torch.Tensor:
        if not images:
            return torch.empty(0, 0)

        def _maybe_normalize(t: torch.Tensor) -> torch.Tensor:
            return _l2_normalize(t) if normalize else t

        out: List[torch.Tensor] = []
        for i in range(0, len(images), max(1, int(batch_size))):
            batch = images[i : i + max(1, int(batch_size))]

            # vLLM multimodal embeddings are chat-shaped: each request embeds one
            # chat with a user message containing a text part and an image_url part.
            # There is no documented batch form for chat-shaped embeddings, so we
            # issue one request per image.
            vecs: List[List[float]] = []
            for img in batch:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "padding"},
                            {"type": "image_url", "image_url": {"url": self._image_to_image_url(img)}},
                        ],
                    }
                ]
                v = self._post_embeddings_chat(messages)
                if not v:
                    raise RuntimeError("vLLM returned empty embeddings for multimodal request")
                vecs.append(v[0])

            out.append(_maybe_normalize(torch.tensor(vecs, dtype=torch.float32)))

        return torch.cat(out, dim=0) if out else torch.empty(0, 0)

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 128, normalize: bool = True) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)

        def _maybe_normalize(t: torch.Tensor) -> torch.Tensor:
            return _l2_normalize(t) if normalize else t

        out: List[torch.Tensor] = []
        if batch_size <= 1:
            for t in texts:
                vecs = self._post_embeddings(t)
                out.append(_maybe_normalize(torch.tensor(vecs, dtype=torch.float32)))
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = self._post_embeddings(batch)
                out.append(_maybe_normalize(torch.tensor(vecs, dtype=torch.float32)))
        return torch.cat(out, dim=0)
