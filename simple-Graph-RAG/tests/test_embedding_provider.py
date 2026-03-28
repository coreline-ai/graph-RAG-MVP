from __future__ import annotations

import asyncio
import sys
import threading
import time
import types

import pytest

from app.adapters.embedding_provider import BgeM3EmbeddingProvider
from app.config import Settings


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


@pytest.mark.asyncio
async def test_embedding_provider_serializes_initial_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor_calls = 0
    concurrent_loads = 0
    max_concurrent_loads = 0
    state_lock = threading.Lock()

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            nonlocal constructor_calls, concurrent_loads, max_concurrent_loads
            with state_lock:
                constructor_calls += 1
                concurrent_loads += 1
                max_concurrent_loads = max(max_concurrent_loads, concurrent_loads)
            time.sleep(0.05)
            with state_lock:
                concurrent_loads -= 1

        def encode(self, texts, batch_size: int, normalize_embeddings: bool):
            return [_FakeVector([0.1, 0.2, 0.3]) for _ in texts]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    provider = BgeM3EmbeddingProvider(Settings())

    first, second = await asyncio.gather(
        provider.embed_texts(["첫 번째"]),
        provider.embed_texts(["두 번째"]),
    )

    assert first == [[0.1, 0.2, 0.3]]
    assert second == [[0.1, 0.2, 0.3]]
    assert constructor_calls == 1
    assert max_concurrent_loads == 1
