"""Endpoint composition tests for OpenAI-compatible LLM providers."""

from __future__ import annotations

import pytest

from memory_agent.integrations.llm_connector import LLMConnector, PROVIDERS


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "synthetic response"}}]}


class FakeHttpClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def post(self, url: str, *, headers: dict, json: dict) -> FakeResponse:
        self.requests.append({"url": url, "headers": headers, "json": json})
        return FakeResponse()


@pytest.mark.parametrize("provider", ["qwencloud", "deepseek", "openrouter", "openai"])
def test_openai_compatible_providers_use_one_chat_completions_v1_path(provider: str):
    """Each provider must compose one canonical /v1/chat/completions endpoint."""
    connector = LLMConnector.__new__(LLMConnector)
    connector.provider = provider
    connector.base_url = PROVIDERS[provider]["base_url"]
    connector.api_key = "test-api-key"
    connector.model = f"{provider}-test-model"

    fake_http = FakeHttpClient()
    connector.http = fake_http
    messages = [{"role": "user", "content": "Say hello"}]

    response = connector._call_llm(messages)

    assert response == "synthetic response"
    assert len(fake_http.requests) == 1
    request = fake_http.requests[0]

    base_url = connector.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    expected_url = f"{base_url}/chat/completions"
    assert request["url"] == expected_url
    assert request["url"].count("/v1/chat/completions") == 1
    assert "/v1/v1/" not in request["url"]
    assert request["json"]["model"] == connector.model
    assert request["json"]["messages"] == messages
