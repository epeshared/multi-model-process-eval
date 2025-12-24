# src/tasks/embedding_backends/sglang_http.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import base64
import mimetypes
import os

import torch

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


class SGLangHTTPEmbeddingClient:
    """
    SGLang HTTP embedding client.

    NOTE:
      - Your SGLang server's profiling endpoints are:
            /start_profile
            /stop_profile
        (NOT under /v1), as you verified via curl.

      - We keep embedding endpoints as-is:
            /encode                 (native)
            /v1/embeddings          (v1 / openai-compatible)

      - We also tag this client as sglang-http so upper layers (tasks/embedding.py)
        can decide to do start/stop around encode() calls (方案A).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api: str = "native",
        api_key: str = "",
        timeout: float = 120.0,
        image_transport: str = "data-url",
    ) -> None:
        if not _REQUESTS_OK and api in {"native", "v1", "openai"}:
            raise RuntimeError("requests is required for the sglang HTTP embedding backend")
        self.base_url = _norm_base_url(base_url)
        self.model = model
        self.api = (api or "native").lower()
        self.api_key = api_key or ""
        self.timeout = float(timeout)
        self.image_transport = (image_transport or "data-url").lower()
        self.session = requests.Session() if _REQUESTS_OK else None

        # ✅ 方案A：给 session 打 tag（上层只根据 tag 才会 start/stop profile）
        self._backend_tag = "sglang-http"

        self._openai_client = None
        if self.api == "openai":
            try:
                import openai  # type: ignore
            except Exception as e:
                raise RuntimeError(f"openai package is required for api='openai': {e}")
            self._openai_client = openai.Client(base_url=f"{self.base_url}/v1", api_key=self.api_key or "None")

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        session = self.session
        if session is None:
            raise RuntimeError("HTTP session is not initialized")
        url = f"{self.base_url}{path}"
        resp = session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {"data": data}

    # ---------------------------
    # Profiling endpoints (root)
    # ---------------------------
    def start_profile(self, **kwargs: Any) -> Dict[str, Any]:
        # default matches your curl: http://host:port/start_profile
        start_path = str(kwargs.pop("start_path", "/start_profile"))
        payload: Dict[str, Any] = dict(kwargs)
        payload.setdefault("model", self.model or "default")
        return self._post_json(start_path, payload)

    def stop_profile(self, **kwargs: Any) -> Dict[str, Any]:
        stop_path = str(kwargs.pop("stop_path", "/stop_profile"))
        payload: Dict[str, Any] = dict(kwargs)
        payload.setdefault("model", self.model or "default")
        return self._post_json(stop_path, payload)

    # ---------------------------
    # Embedding endpoints
    # ---------------------------
    def _encode_native_one(self, text: str) -> List[float]:
        url = f"{self.base_url}/encode"
        session = self.session
        if session is None:
            raise RuntimeError("HTTP session is not initialized")
        payload: Dict[str, Any] = {"text": text}
        if self.model:
            payload["model"] = self.model
        resp = session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "embedding" in data:
            return list(map(float, data["embedding"]))
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list) and data["data"]:
            return list(map(float, data["data"][0]["embedding"]))
        raise RuntimeError(f"Unexpected response from /encode: {data}")

    def _encode_v1_any(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        url = f"{self.base_url}/v1/embeddings"
        session = self.session
        if session is None:
            raise RuntimeError("HTTP session is not initialized")
        payload: Dict[str, Any] = {
            "model": self.model or "default",
            "input": inputs,
            # NOTE: keep your existing behavior; caller can override if needed by editing later
            "encoding_format": "bf16",
        }
        resp = session.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
            raise RuntimeError(f"Unexpected response from /v1/embeddings: {data}")
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [list(map(float, r["embedding"])) for r in rows]

    def _encode_v1_multimodal_any(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[List[float]]:
        url = f"{self.base_url}/v1/embeddings"
        session = self.session
        if session is None:
            raise RuntimeError("HTTP session is not initialized")
        payload: Dict[str, Any] = {
            "model": self.model or "default",
            "input": inputs,
        }
        resp = session.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
            raise RuntimeError(f"Unexpected response from /v1/embeddings: {data}")
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [list(map(float, r["embedding"])) for r in rows]

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
                mime = mime or "application/octet-stream"
                with open(s, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                return f"data:{mime};base64,{b64}"
            return s

        if isinstance(img, (bytes, bytearray)):
            b64 = base64.b64encode(bytes(img)).decode("ascii")
            return f"data:application/octet-stream;base64,{b64}"

        return img

    @torch.inference_mode()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 128,
        normalize: bool = True,
        # profile passthrough (ignored here; start/stop should be in tasks/embedding.py per 方案A)
        profile: bool = False,
        profile_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        _ = profile
        _ = profile_kwargs

        if not texts:
            return torch.empty(0, 0)
        out: List[torch.Tensor] = []

        if self.api == "openai":
            client = self._openai_client
            if client is None:
                raise RuntimeError("openai client is not initialized")
            for i in range(0, len(texts), max(1, int(batch_size))):
                batch = texts[i : i + max(1, int(batch_size))]
                inp: Union[str, List[str]] = batch[0] if len(batch) == 1 else batch
                resp = client.embeddings.create(model=(self.model or "default"), input=inp)
                rows = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                vecs = [list(map(float, r.embedding)) for r in rows]
                emb = torch.tensor(vecs, dtype=torch.float32)
                out.append(torch.nn.functional.normalize(emb, p=2, dim=1) if normalize else emb)
            hidden = int(out[0].shape[-1]) if out and out[0].numel() > 0 else 0
            return torch.cat(out, dim=0) if out else torch.empty(0, hidden)

        if self.api == "v1":
            for i in range(0, len(texts), max(1, int(batch_size))):
                batch = texts[i : i + max(1, int(batch_size))]
                inp = batch[0] if len(batch) == 1 else batch
                vecs = self._encode_v1_any(inp)
                emb = torch.tensor(vecs, dtype=torch.float32)
                out.append(torch.nn.functional.normalize(emb, p=2, dim=1) if normalize else emb)
            hidden = int(out[0].shape[-1]) if out and out[0].numel() > 0 else 0
            return torch.cat(out, dim=0) if out else torch.empty(0, hidden)

        # native
        for i in range(0, len(texts), max(1, int(batch_size))):
            batch = texts[i : i + max(1, int(batch_size))]
            vecs = [self._encode_native_one(s) for s in batch]
            emb = torch.tensor(vecs, dtype=torch.float32)
            out.append(torch.nn.functional.normalize(emb, p=2, dim=1) if normalize else emb)
        hidden = int(out[0].shape[-1]) if out and out[0].numel() > 0 else 0
        return torch.cat(out, dim=0) if out else torch.empty(0, hidden)

    @torch.inference_mode()
    def encode_images(
        self,
        images: List[Any],
        batch_size: int = 128,
        normalize: bool = True,
        # profile passthrough (ignored here; start/stop should be in tasks/embedding.py per 方案A)
        profile: bool = False,
        profile_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        _ = profile
        _ = profile_kwargs

        if not images:
            return torch.empty(0, 0)
        if self.api not in {"v1", "openai"}:
            raise RuntimeError("Image embeddings require api='v1' or api='openai'")

        out: List[torch.Tensor] = []
        probed = False
        for i in range(0, len(images), max(1, int(batch_size))):
            batch = images[i : i + max(1, int(batch_size))]
            inputs = [{"text": "padding", "image": self._image_to_repr(x)} for x in batch]

            if not probed and inputs:
                probed = True
                try:
                    probe_inputs = [inputs[0], {"text": "padding", "image": "/no/such/file.jpg"}]
                    probe_vecs = self._encode_v1_multimodal_any(probe_inputs)
                    if len(probe_vecs) >= 2:
                        a = torch.tensor(probe_vecs[0], dtype=torch.float32)
                        b = torch.tensor(probe_vecs[1], dtype=torch.float32)
                        if torch.allclose(a, b, atol=0.0, rtol=0.0):
                            raise RuntimeError(
                                "SGLang server appears to ignore image inputs for /v1/embeddings (multimodal)."
                            )
                except RuntimeError:
                    raise
                except Exception:
                    pass

            vecs = self._encode_v1_multimodal_any(inputs)
            emb = torch.tensor(vecs, dtype=torch.float32)
            out.append(torch.nn.functional.normalize(emb, p=2, dim=1) if normalize else emb)

        hidden = int(out[0].shape[-1]) if out and out[0].numel() > 0 else 0
        return torch.cat(out, dim=0) if out else torch.empty(0, hidden)
