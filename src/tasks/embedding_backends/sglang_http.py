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


class SGLangHTTPEmbeddingClient:
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
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api = api.lower()
        self.api_key = api_key or ""
        self.timeout = timeout
        self.image_transport = (image_transport or "data-url").lower()
        self.session = requests.Session() if _REQUESTS_OK else None

        self._openai_client = None
        if self.api == "openai":
            try:
                import openai  # type: ignore
            except Exception as e:
                raise RuntimeError(f"openai package is required for api='openai': {e}")
            self._openai_client = openai.Client(base_url=f"{self.base_url}/v1", api_key=self.api_key or "None")

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
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": self.model or "default",
            "input": inputs,
            "encoding_format": "bf16",
        }
        resp = session.post(url, json=payload, headers=headers, timeout=self.timeout)
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
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": self.model or "default",
            "input": inputs,
        }
        resp = session.post(url, json=payload, headers=headers, timeout=self.timeout)
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
    def encode(self, texts: List[str], batch_size: int = 128, normalize: bool = True) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)
        out: List[torch.Tensor] = []

        if self.api == "openai":
            client = self._openai_client
            if client is None:
                raise RuntimeError("openai client is not initialized")
            if batch_size <= 1:
                for text in texts:
                    resp = client.embeddings.create(model=(self.model or "default"), input=text)
                    rows = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                    vecs = [list(map(float, r.embedding)) for r in rows]
                    emb = torch.tensor(vecs, dtype=torch.float32)
                    out.append(torch.nn.functional.normalize(emb, p=2, dim=1) if normalize else emb)
            else:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    resp = client.embeddings.create(model=(self.model or "default"), input=batch)
                    rows = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                    vecs = [list(map(float, r.embedding)) for r in rows]
                    emb = torch.tensor(vecs, dtype=torch.float32)
                    out.append(torch.nn.functional.normalize(emb, p=2, dim=1) if normalize else emb)
            hidden = len(out[0][0]) if out and out[0].numel() > 0 else 0
            return torch.cat(out, dim=0) if out else torch.empty(0, hidden)

        if self.api == "v1":
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = self._encode_v1_any(batch if batch_size > 1 else batch[0])
                emb = torch.tensor(vecs, dtype=torch.float32)
                out.append(torch.nn.functional.normalize(emb, p=2, dim=1) if normalize else emb)
            hidden = len(out[0][0]) if out and out[0].numel() > 0 else 0
            return torch.cat(out, dim=0) if out else torch.empty(0, hidden)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = [self._encode_native_one(s) for s in batch]
            emb = torch.tensor(vecs, dtype=torch.float32)
            out.append(torch.nn.functional.normalize(emb, p=2, dim=1) if normalize else emb)
        hidden = len(out[0][0]) if out and out[0].numel() > 0 else 0
        return torch.cat(out, dim=0) if out else torch.empty(0, hidden)

    @torch.inference_mode()
    def encode_images(self, images: List[Any], batch_size: int = 128, normalize: bool = True) -> torch.Tensor:
        if not images:
            return torch.empty(0, 0)
        if self.api not in {"v1", "openai"}:
            raise RuntimeError("Image embeddings require api='v1' or api='openai'")

        out: List[torch.Tensor] = []
        probed = False
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
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

        hidden = len(out[0][0]) if out and out[0].numel() > 0 else 0
        return torch.cat(out, dim=0) if out else torch.empty(0, hidden)
