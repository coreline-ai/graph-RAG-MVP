from __future__ import annotations

import httpx

from app.config import Settings
from app.schemas import CodexGenerateRequest, CodexGenerateResponse


class CodexProxyError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class CodexProxyClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.codex_proxy_base_url,
            timeout=settings.codex_timeout_seconds,
        )

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, object] | None = None,
    ) -> CodexGenerateResponse:
        payload = CodexGenerateRequest(
            model=self.settings.codex_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata or {},
        )
        try:
            response = await self._client.post("/generate", json=payload.model_dump())
        except httpx.TimeoutException as exc:
            raise CodexProxyError("llm_unavailable", str(exc)) from exc
        except httpx.HTTPError as exc:
            raise CodexProxyError("llm_unavailable", str(exc)) from exc

        if response.status_code in {401, 403}:
            raise CodexProxyError("reauth_required", response.text)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CodexProxyError("llm_unavailable", response.text) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise CodexProxyError("llm_protocol_error", response.text) from exc

        text = body.get("text")
        if not isinstance(text, str):
            raw_message = str(body)
            if "reauth" in raw_message.lower():
                raise CodexProxyError("reauth_required", raw_message)
            raise CodexProxyError("llm_protocol_error", raw_message)

        return CodexGenerateResponse(
            text=text,
            model=body.get("model"),
            finish_reason=body.get("finish_reason"),
            raw=body,
        )

    async def healthcheck(self) -> str:
        try:
            response = await self._client.get("/health")
        except httpx.HTTPError as exc:
            return f"error:{exc}"
        if response.status_code < 500:
            return "ok"
        return f"error:{response.status_code}"

    async def aclose(self) -> None:
        await self._client.aclose()
