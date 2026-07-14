"""Contract test for Anthropic's native Messages API request and response shape."""

from __future__ import annotations

from memory_agent.integrations.llm_connector import LLMConnector, PROVIDERS


class FakeAnthropicResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"content": [{"type": "text", "text": "ok"}]}


class RecordingAnthropicHttpClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def post(self, url: str, *, headers: dict, json: dict) -> FakeAnthropicResponse:
        self.requests.append({"url": url, "headers": headers, "json": json})
        return FakeAnthropicResponse()


def test_anthropic_connector_uses_native_messages_api_contract():
    """Anthropic calls use /v1/messages, native headers, and split system content."""
    connector = LLMConnector.__new__(LLMConnector)
    connector.provider = "anthropic"
    connector.base_url = PROVIDERS["anthropic"]["base_url"]
    connector.api_key = "anthropic-test-key"
    connector.model = "claude-test-model"
    fake_http = RecordingAnthropicHttpClient()
    connector.http = fake_http

    response = connector._call_llm(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hello."},
        ]
    )

    assert response == "ok"
    assert len(fake_http.requests) == 1
    request = fake_http.requests[0]

    assert request["url"].endswith("/v1/messages")
    assert "/chat/completions" not in request["url"]
    assert request["headers"]["x-api-key"] == "anthropic-test-key"
    assert request["headers"]["anthropic-version"] == "2023-06-01"
    assert request["json"]["model"] == "claude-test-model"
    assert request["json"]["system"] == "You are concise."
    assert request["json"]["messages"] == [
        {"role": "user", "content": "Say hello."}
    ]
    assert all(message["role"] != "system" for message in request["json"]["messages"])
