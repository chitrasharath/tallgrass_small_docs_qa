"""Embedding models: a local ONNX model (fastembed) and the 4Geeks LiteLLM proxy."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Protocol

import httpx
import numpy as np

from doc_qa.config import llm_settings

LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    model_name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingError(Exception):
    pass


class LocalEmbedder:
    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.passage_embed(texts)))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.query_embed(texts)))


_RESET_AT = re.compile(r"Limit resets at:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*UTC", re.I)


def _rate_limit_wait(body: str) -> float:
    """Seconds until the proxy's rate limit resets, from its error message (default 65)."""
    match = _RESET_AT.search(body)
    if match:
        reset = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return min(max((reset - datetime.now(timezone.utc)).total_seconds() + 1, 1.0), 120.0)
    return 65.0


class ApiEmbedder:
    """OpenAI-compatible /v1/embeddings on the 4Geeks LiteLLM proxy.

    Texts are sent in batches because the proxy allows only about 15 requests a minute.
    """

    BATCH_SIZE = 64
    RATE_LIMIT_RETRIES = 6
    SERVER_ERROR_RETRIES = 1

    def __init__(self, model_name: str | None = None):
        settings = llm_settings()
        if not settings.api_key:
            raise EmbeddingError("LLM_API_KEY is not set. Add it to .env (see .example.env).")
        self.model_name = model_name or settings.embedding_model
        self._url = f"{settings.base_url.rstrip('/')}/v1/embeddings"
        self._headers = {"Authorization": f"Bearer {settings.api_key}"}
        self._verify = settings.ssl_verify

    def _post(self, batch: list[str]) -> list[list[float]]:
        server_errors = 0
        for _ in range(self.RATE_LIMIT_RETRIES + 1):
            try:
                with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0), verify=self._verify) as client:
                    response = client.post(
                        self._url, headers=self._headers, json={"model": self.model_name, "input": batch}
                    )
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc
            if response.status_code == 429:
                time.sleep(_rate_limit_wait(response.text))
                continue
            if response.status_code >= 500 and server_errors < self.SERVER_ERROR_RETRIES:
                server_errors += 1
                continue
            if not response.is_success:
                raise EmbeddingError(f"Embedding proxy returned {response.status_code}: {response.text[:300]}")
            data = response.json().get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise EmbeddingError("Malformed embedding response: expected one vector per input")
            return [item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0))]
        raise EmbeddingError("Embedding proxy kept rate limiting; try again later")

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            vectors += self._post(texts[start : start + self.BATCH_SIZE])
        return np.array(vectors, dtype=float)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)
