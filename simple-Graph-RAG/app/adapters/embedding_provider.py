from __future__ import annotations

import asyncio
from typing import Protocol

from app.config import Settings

_MAX_LOAD_ATTEMPTS = 3


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def healthcheck(self) -> str: ...


class BgeM3EmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._last_error: str | None = None
        self._load_attempts: int = 0

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._load_attempts >= _MAX_LOAD_ATTEMPTS:
            raise RuntimeError(
                f"Embedding model loading failed {self._load_attempts} times. "
                f"Last error: {self._last_error}"
            )
        self._load_attempts += 1
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import path depends on env
            self._last_error = str(exc)
            raise
        try:
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
            )
        except Exception as exc:
            self._last_error = str(exc)
            raise
        self._load_attempts = 0
        return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)

    async def healthcheck(self) -> str:
        if self._load_attempts >= _MAX_LOAD_ATTEMPTS:
            return f"error:circuit_open:{self._last_error}"
        if self._last_error and self._model is None:
            return f"error:{self._last_error}"
        if self._model is not None:
            return "ok"
        return "not_loaded"
