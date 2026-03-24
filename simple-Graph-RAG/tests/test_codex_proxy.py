from __future__ import annotations

import httpx
import pytest

from app.adapters.codex_proxy import CodexProxyClient, CodexProxyError
from app.config import Settings


@pytest.mark.asyncio
async def test_codex_proxy_maps_timeout_to_unavailable() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timeout")

    client = CodexProxyClient(Settings())
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    )

    with pytest.raises(CodexProxyError, match="timeout") as exc_info:
        await client.generate(system_prompt="system", user_prompt="user")

    await client.aclose()
    assert exc_info.value.code == "llm_unavailable"


@pytest.mark.asyncio
async def test_codex_proxy_maps_forbidden_to_reauth_required() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=403, text="reauth required")

    client = CodexProxyClient(Settings())
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    )

    with pytest.raises(CodexProxyError, match="reauth required") as exc_info:
        await client.generate(system_prompt="system", user_prompt="user")

    await client.aclose()
    assert exc_info.value.code == "reauth_required"
