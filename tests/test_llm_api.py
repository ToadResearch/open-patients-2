import asyncio
import os
import types
import unittest
from unittest.mock import patch

from src.core import llm_api
from src.core.llm_api import APISettings, ChatRequest, EndpointConfig, request_chat_completion


class RetryableStatusError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"status={code}")
        self.status_code = code


def _fake_response(
    text: str = "{}",
    prompt_tokens: int = 10,
    completion_tokens: int = 3,
    reasoning: str | None = None,
    reasoning_content: str | None = None,
):
    usage = types.SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = types.SimpleNamespace(
        content=text,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
    )
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class _FakeCompletions:
    def __init__(self, items):
        self._items = list(items)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, items):
        self.completions = _FakeCompletions(items)
        self.chat = types.SimpleNamespace(completions=self.completions)


class LLMApiTests(unittest.TestCase):
    def test_default_client_factory_requires_api_key_env_for_remote(self) -> None:
        endpoint = EndpointConfig(
            name="missing-key",
            base_url="https://api.example.com/v1",
            model="test-model",
            api_key_env="OP_MISSING_TEST_KEY",
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Set environment variable OP_MISSING_TEST_KEY"):
                llm_api._default_client_factory(endpoint, 30.0)

    def test_default_client_factory_uses_dummy_key_for_localhost(self) -> None:
        endpoint = EndpointConfig(
            name="local",
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            api_key_env="OP_MISSING_TEST_KEY",
        )

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.core.llm_api.AsyncOpenAI") as mock_client:
                llm_api._default_client_factory(endpoint, 30.0)

        mock_client.assert_called_once_with(
            base_url="http://127.0.0.1:8000/v1",
            api_key="dummy",
            timeout=30.0,
        )

    def test_request_chat_completion_uses_json_schema_mode(self) -> None:
        endpoint = EndpointConfig(
            name="local",
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            structured_mode="json_schema",
        )
        req = ChatRequest(
            request_id="r1",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"x": 1},
        )
        client = _FakeClient([_fake_response("{}")])

        result = asyncio.run(
            request_chat_completion(
                client=client,
                endpoint=endpoint,
                api_settings=APISettings(max_retries=0),
                request=req,
                sampling_cfg={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 32,
                    "seed": 7,
                },
                structured_output=True,
                json_schema={"json_schema": {"schema": {"type": "object"}}},
            )
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.text, "{}")
        self.assertEqual(result.usage["prompt_tokens"], 10)
        call = client.completions.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["max_tokens"], 32)
        self.assertEqual(call["response_format"]["type"], "json_schema")

    def test_request_chat_completion_captures_reasoning_field(self) -> None:
        endpoint = EndpointConfig(
            name="local",
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            structured_mode="none",
        )
        req = ChatRequest(
            request_id="r1",
            messages=[{"role": "user", "content": "hello"}],
        )
        client = _FakeClient([_fake_response(text='{"x":1}', reasoning="internal reasoning text")])

        result = asyncio.run(
            request_chat_completion(
                client=client,
                endpoint=endpoint,
                api_settings=APISettings(max_retries=0),
                request=req,
                sampling_cfg={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 32},
                structured_output=False,
                json_schema=None,
            )
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.reasoning, "internal reasoning text")

    def test_request_chat_completion_captures_reasoning_content_field(self) -> None:
        endpoint = EndpointConfig(
            name="local",
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            structured_mode="none",
        )
        req = ChatRequest(
            request_id="r1",
            messages=[{"role": "user", "content": "hello"}],
        )
        client = _FakeClient(
            [_fake_response(text='{"x":1}', reasoning_content="parser reasoning content")]
        )

        result = asyncio.run(
            request_chat_completion(
                client=client,
                endpoint=endpoint,
                api_settings=APISettings(max_retries=0),
                request=req,
                sampling_cfg={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 32},
                structured_output=False,
                json_schema=None,
            )
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.reasoning, "parser reasoning content")

    def test_request_chat_completion_retries_retryable_status(self) -> None:
        endpoint = EndpointConfig(
            name="local",
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            structured_mode="none",
        )
        req = ChatRequest(
            request_id="r1",
            messages=[{"role": "user", "content": "hello"}],
        )
        client = _FakeClient([RetryableStatusError(429), _fake_response("ok")])

        result = asyncio.run(
            request_chat_completion(
                client=client,
                endpoint=endpoint,
                api_settings=APISettings(max_retries=2, retry_backoff_initial_s=0.0),
                request=req,
                sampling_cfg={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 16},
                structured_output=False,
                json_schema=None,
            )
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(client.completions.calls), 2)

    def test_request_chat_completion_omits_max_tokens_when_unset(self) -> None:
        endpoint = EndpointConfig(
            name="local",
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            structured_mode="none",
        )
        req = ChatRequest(
            request_id="r1",
            messages=[{"role": "user", "content": "hello"}],
        )
        client = _FakeClient([_fake_response("ok")])

        result = asyncio.run(
            request_chat_completion(
                client=client,
                endpoint=endpoint,
                api_settings=APISettings(max_retries=0),
                request=req,
                sampling_cfg={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": None},
                structured_output=False,
                json_schema=None,
            )
        )

        self.assertIsNone(result.error)
        call = client.completions.calls[0]
        self.assertNotIn("max_tokens", call)

    def test_request_chat_completion_omits_max_tokens_when_non_positive(self) -> None:
        endpoint = EndpointConfig(
            name="local",
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            structured_mode="none",
        )
        req = ChatRequest(
            request_id="r1",
            messages=[{"role": "user", "content": "hello"}],
        )
        client = _FakeClient([_fake_response("ok")])

        result = asyncio.run(
            request_chat_completion(
                client=client,
                endpoint=endpoint,
                api_settings=APISettings(max_retries=0),
                request=req,
                sampling_cfg={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": -1},
                structured_output=False,
                json_schema=None,
            )
        )

        self.assertIsNone(result.error)
        call = client.completions.calls[0]
        self.assertNotIn("max_tokens", call)


if __name__ == "__main__":
    unittest.main()
