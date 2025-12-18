from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import time

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

    def _post_embeddings(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload: Dict[str, Any] = {"model": self.model, "input": inputs}
        if self.encoding_format:
            payload["encoding_format"] = self.encoding_format

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                rows = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                return [list(map(float, r["embedding"])) for r in rows]
            except Exception as e:  # pragma: no cover - network
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"vLLM HTTP embeddings failed after {self.max_retries} attempts: {last_err}")

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)
        out: List[torch.Tensor] = []
        if batch_size <= 1:
            for t in texts:
                vecs = self._post_embeddings(t)
                out.append(_l2_normalize(torch.tensor(vecs, dtype=torch.float32)))
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = self._post_embeddings(batch)
                out.append(_l2_normalize(torch.tensor(vecs, dtype=torch.float32)))
        return torch.cat(out, dim=0)
