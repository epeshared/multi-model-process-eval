from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import time
import base64
import sys
from array import array

import torch

try:
    import requests  # type: ignore
    _REQUESTS_OK = True
except Exception:
    _REQUESTS_OK = False


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

        self.session = requests.Session()

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
