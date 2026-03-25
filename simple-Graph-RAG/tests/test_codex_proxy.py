from __future__ import annotations

import json

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


@pytest.mark.asyncio
async def test_codex_proxy_auto_falls_back_to_openai_responses() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/generate":
            return httpx.Response(status_code=404, text="not found")
        if request.url.path == "/openai/v1/responses":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload["model"] == "gpt-5.3-codex"
            assert payload["input"][0]["role"] == "system"
            assert payload["input"][0]["content"][0]["text"] == "system"
            assert payload["input"][1]["role"] == "user"
            assert payload["input"][1]["content"][0]["text"] == "user"
            return httpx.Response(
                status_code=200,
                json={
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "proxy ok"}],
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = CodexProxyClient(Settings())
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    )

    response = await client.generate(system_prompt="system", user_prompt="user")

    await client.aclose()
    assert response.text == "proxy ok"
    assert client._resolved_api_style == "openai_responses"


@pytest.mark.asyncio
async def test_codex_proxy_healthcheck_supports_multi_model_tui_route() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(status_code=404, text="not found")
        if request.url.path == "/api/v1/health":
            return httpx.Response(status_code=200, json={"ok": True})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = CodexProxyClient(Settings())
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    )

    status = await client.healthcheck()

    await client.aclose()
    assert status == "ok"
    assert client._resolved_api_style == "openai_responses"
