from __future__ import annotations

from typing import Any, Literal

import httpx

from app.config import Settings
from app.schemas import CodexGenerateRequest, CodexGenerateResponse


class CodexProxyError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


ProxyApiStyle = Literal["auto", "legacy", "openai_responses", "openai_chat"]


_LOCAL_MODELS = {"Qwen3_5-9B-IQ4_XS"}


class CodexProxyClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._configured_api_style: ProxyApiStyle = settings.codex_proxy_api_style
        self._resolved_api_style: ProxyApiStyle | None = (
            None if settings.codex_proxy_api_style == "auto" else settings.codex_proxy_api_style
        )
        self._local_client = httpx.AsyncClient(
            base_url=settings.codex_proxy_base_url,
            timeout=settings.codex_timeout_seconds,
        )
        self._client = self._local_client
        self._claude_client = httpx.AsyncClient(
            base_url=getattr(settings, "claude_proxy_base_url", "http://127.0.0.1:8800"),
            timeout=settings.codex_timeout_seconds,
        )

    def _is_local_model(self, model: str) -> bool:
        return model in _LOCAL_MODELS

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_override: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CodexGenerateResponse:
        model = model_override or self.settings.codex_model
        payload = CodexGenerateRequest(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata or {},
        )

        if self._is_local_model(model):
            return await self._generate_openai_chat(payload)

        return await self._generate_via_claude_proxy(payload)

    async def _generate_via_claude_proxy(
        self, payload: CodexGenerateRequest
    ) -> CodexGenerateResponse:
        body = {
            "system_prompt": payload.system_prompt,
            "user_prompt": payload.user_prompt,
            "model": payload.model,
        }
        try:
            response = await self._claude_client.post("/generate", json=body)
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            raise CodexProxyError("llm_unavailable", str(exc)) from exc
        if response.status_code >= 400:
            raise CodexProxyError("llm_unavailable", response.text)
        parsed = response.json()
        return self._parse_legacy_response(parsed)

    async def healthcheck(self) -> str:
        errors = []
        # Check Claude Proxy
        try:
            r = await self._claude_client.get("/health")
            if r.status_code >= 500:
                errors.append(f"claude_proxy:{r.status_code}")
        except httpx.HTTPError as exc:
            errors.append(f"claude_proxy:{exc}")
        # Check Local LLM
        try:
            r = await self._local_client.get("/v1/models")
            if r.status_code >= 500:
                errors.append(f"local_llm:{r.status_code}")
        except httpx.HTTPError as exc:
            errors.append(f"local_llm:{exc}")
        if not errors:
            return "ok"
        if len(errors) == 2:
            return f"error:{'; '.join(errors)}"
        return f"partial:{errors[0]}"

    def _candidate_api_styles(self) -> tuple[ProxyApiStyle, ...]:
        if self._resolved_api_style is not None:
            return (self._resolved_api_style,)
        if self._configured_api_style != "auto":
            return (self._configured_api_style,)
        return ("openai_chat", "legacy", "openai_responses")

    async def _generate_legacy(self, payload: CodexGenerateRequest) -> CodexGenerateResponse:
        body = await self._post_json("/generate", payload.model_dump())
        return self._parse_legacy_response(body)

    async def _generate_openai_chat(
        self,
        payload: CodexGenerateRequest,
    ) -> CodexGenerateResponse:
        body = await self._post_json(
            "/chat/completions",
            {
                "model": payload.model,
                "messages": [
                    {"role": "system", "content": payload.system_prompt},
                    {"role": "user", "content": payload.user_prompt},
                ],
                "max_tokens": 4096,
            },
        )
        return self._parse_openai_chat_response(body)

    async def _generate_openai_responses(
        self,
        payload: CodexGenerateRequest,
    ) -> CodexGenerateResponse:
        body = await self._post_json(
            "/openai/v1/responses",
            {
                "model": payload.model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": payload.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": payload.user_prompt}],
                    },
                ],
                "stream": False,
            },
        )
        return self._parse_openai_responses_body(body)

    async def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=body)
        except httpx.TimeoutException as exc:
            raise CodexProxyError("llm_unavailable", str(exc)) from exc
        except httpx.HTTPError as exc:
            raise CodexProxyError("llm_unavailable", str(exc)) from exc

        if response.status_code in {404, 405}:
            raise CodexProxyError(
                "protocol_mismatch",
                f"proxy endpoint {path} is not available at {self.settings.codex_proxy_base_url}",
            )
        if response.status_code in {401, 403}:
            raise CodexProxyError("reauth_required", response.text)
        if response.status_code == 409:
            raise CodexProxyError("human_input_required", response.text)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CodexProxyError("llm_unavailable", response.text) from exc

        try:
            parsed = response.json()
        except ValueError as exc:
            raise CodexProxyError("llm_protocol_error", response.text) from exc

        if not isinstance(parsed, dict):
            raise CodexProxyError("llm_protocol_error", str(parsed))
        return parsed

    @staticmethod
    def _parse_legacy_response(body: dict[str, Any]) -> CodexGenerateResponse:
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

    @staticmethod
    def _parse_openai_chat_response(body: dict[str, Any]) -> CodexGenerateResponse:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise CodexProxyError("llm_protocol_error", str(body))
        message = choices[0].get("message", {})
        text = message.get("content") or ""
        if not text:
            raise CodexProxyError("llm_protocol_error", f"empty content: {body}")
        return CodexGenerateResponse(
            text=text,
            model=body.get("model"),
            finish_reason=choices[0].get("finish_reason"),
            raw=body,
        )

    @staticmethod
    def _parse_openai_responses_body(body: dict[str, Any]) -> CodexGenerateResponse:
        text_fragments: list[str] = []
        output = body.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "message":
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                        text_fragments.append(part["text"])

        text = "".join(text_fragments)
        if not text and isinstance(body.get("output_text"), str):
            text = body["output_text"]
        if not text:
            raw_message = str(body)
            if "reauth" in raw_message.lower():
                raise CodexProxyError("reauth_required", raw_message)
            raise CodexProxyError("llm_protocol_error", raw_message)

        return CodexGenerateResponse(
            text=text,
            model=body.get("model"),
            finish_reason="stop",
            raw=body,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._claude_client.aclose()
