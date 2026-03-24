from __future__ import annotations

import asyncio
from typing import Protocol

from app.config import Settings


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def healthcheck(self) -> str: ...


class BgeM3EmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._last_error: str | None = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - import path depends on env
                self._last_error = str(exc)
                raise
            self._model = SentenceTransformer(self.settings.embedding_model)
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
        if self._last_error:
            return f"error:{self._last_error}"
        if self._model is not None:
            return "ok"
        return "not_loaded"

