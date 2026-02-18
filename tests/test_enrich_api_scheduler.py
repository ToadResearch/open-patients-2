import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import enrich
from src.core.llm_api import (
    APISettings,
    ChatRequest,
    ChatResult,
    EndpointConfig,
    OutageError,
    run_chat_requests,
)


class StatusError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"status={code}")
        self.status_code = code


def _fake_response(text: str = "{}"):
    usage = types.SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3)
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class _DelayCompletions:
    def __init__(self, delay_s: float, fail: bool = False):
        self.delay_s = delay_s
        self.fail = fail

    async def create(self, **_kwargs):
        await asyncio.sleep(self.delay_s)
        if self.fail:
            raise StatusError(503)
        return _fake_response("{}")


class _DelayClient:
    def __init__(self, delay_s: float, fail: bool = False):
        self.chat = types.SimpleNamespace(completions=_DelayCompletions(delay_s, fail=fail))


class SchedulerTests(unittest.TestCase):
    def test_dynamic_queue_prefers_faster_endpoint(self) -> None:
        endpoints = [
            EndpointConfig(name="fast", base_url="http://a/v1", model="m1", concurrency=1),
            EndpointConfig(name="slow", base_url="http://b/v1", model="m2", concurrency=1),
        ]
        requests = [
            ChatRequest(request_id=f"r{i}", messages=[{"role": "user", "content": "x"}])
            for i in range(10)
        ]

        def client_factory(endpoint: EndpointConfig, _timeout_s: float):
            if endpoint.name == "fast":
                return _DelayClient(0.001, fail=False)
            return _DelayClient(0.03, fail=False)

        _, stats = asyncio.run(
            run_chat_requests(
                requests=requests,
                endpoints=endpoints,
                api_settings=APISettings(max_retries=0, outage_abort_after_s=60.0),
                sampling_cfg={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 16},
                structured_output=False,
                json_schema=None,
                client_factory=client_factory,
                queue_size=4,
            )
        )

        fast_submitted = stats["endpoint_stats"]["fast"]["submitted"]
        slow_submitted = stats["endpoint_stats"]["slow"]["submitted"]
        self.assertGreater(fast_submitted, slow_submitted)
        self.assertEqual(stats["completed"], 10)

    def test_outage_abort_when_all_endpoints_unhealthy(self) -> None:
        endpoints = [
            EndpointConfig(name="a", base_url="http://a/v1", model="m1", concurrency=1),
            EndpointConfig(name="b", base_url="http://b/v1", model="m2", concurrency=1),
        ]
        requests = [
            ChatRequest(request_id=f"r{i}", messages=[{"role": "user", "content": "x"}])
            for i in range(100)
        ]

        def client_factory(_endpoint: EndpointConfig, _timeout_s: float):
            return _DelayClient(0.05, fail=True)

        with self.assertRaises(OutageError):
            asyncio.run(
                run_chat_requests(
                    requests=requests,
                    endpoints=endpoints,
                    api_settings=APISettings(
                        max_retries=0,
                        outage_abort_after_s=0.2,
                        retry_backoff_initial_s=0.0,
                        retry_backoff_max_s=0.0,
                    ),
                    sampling_cfg={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 16},
                    structured_output=False,
                    json_schema=None,
                    client_factory=client_factory,
                    queue_size=2,
                )
            )


class EnrichFailureArtifactsTests(unittest.TestCase):
    def test_failed_ids_and_failed_records_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            args = enrich.parse_args(
                [
                    "--out_dir",
                    str(out_dir),
                    "--run_id",
                    "run_test",
                    "--api_base_url",
                    "http://127.0.0.1:8000/v1",
                    "--model",
                    "fake-model",
                    "--max_notes",
                    "2",
                    "--max_new_tokens",
                    "16",
                ]
            )
            rows = [
                {"_id": "id-fail", "description": "note one"},
                {"_id": "id-ok", "description": "note two"},
            ]

            async def fake_run_chat_requests(*, requests, on_result, **_kwargs):
                reqs = list(requests)
                for req in reqs:
                    if req.request_id == "id-fail":
                        res = ChatResult(
                            request_id=req.request_id,
                            endpoint_name="fake",
                            text="",
                            reasoning=None,
                            usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
                            attempts=2,
                            error="StatusError: status=503",
                            metadata=req.metadata,
                        )
                    else:
                        res = ChatResult(
                            request_id=req.request_id,
                            endpoint_name="fake",
                            text="{}",
                            reasoning=None,
                            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                            attempts=1,
                            error=None,
                            metadata=req.metadata,
                        )
                    await on_result(res)
                return [], {
                    "submitted": len(reqs),
                    "completed": len(reqs),
                    "outage_abort": False,
                    "endpoint_stats": {
                        "fake": {"submitted": len(reqs), "succeeded": 1, "failed": 1}
                    },
                }

            with patch("src.cli.enrich.parse_args", return_value=args), patch(
                "src.cli.enrich.load_dataset", return_value=rows
            ), patch("src.cli.enrich.run_chat_requests", new=fake_run_chat_requests):
                enrich.main()

            run_dir = out_dir / "run_test"
            failed_ids_path = run_dir / "failed_ids.txt"
            self.assertTrue(failed_ids_path.exists())
            self.assertIn("id-fail", failed_ids_path.read_text(encoding="utf-8"))

            failed_shards = sorted((run_dir / "shards").glob("failed_records_*.jsonl"))
            self.assertTrue(failed_shards)
            first_failed = json.loads(failed_shards[0].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_failed["id"], "id-fail")
            self.assertEqual(first_failed["reason"], "request_error")

    def test_reasoning_is_saved_and_parse_uses_final_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            args = enrich.parse_args(
                [
                    "--out_dir",
                    str(out_dir),
                    "--run_id",
                    "run_reasoning",
                    "--api_base_url",
                    "http://127.0.0.1:8000/v1",
                    "--model",
                    "fake-model",
                    "--max_notes",
                    "1",
                    "--max_new_tokens",
                    "16",
                ]
            )
            rows = [{"_id": "id-1", "description": "note one"}]

            async def fake_run_chat_requests(*, requests, on_result, **_kwargs):
                reqs = list(requests)
                for req in reqs:
                    res = ChatResult(
                        request_id=req.request_id,
                        endpoint_name="fake",
                        text="<think>hidden chain of thought</think>{\"outcome\": null}",
                        reasoning=None,
                        usage={"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
                        attempts=1,
                        error=None,
                        metadata=req.metadata,
                    )
                    await on_result(res)
                return [], {
                    "submitted": len(reqs),
                    "completed": len(reqs),
                    "outage_abort": False,
                    "endpoint_stats": {
                        "fake": {"submitted": len(reqs), "succeeded": 1, "failed": 0}
                    },
                }

            with patch("src.cli.enrich.parse_args", return_value=args), patch(
                "src.cli.enrich.load_dataset", return_value=rows
            ), patch("src.cli.enrich.run_chat_requests", new=fake_run_chat_requests):
                enrich.main()

            run_dir = out_dir / "run_reasoning"
            data_path = run_dir / "data.jsonl"
            self.assertTrue(data_path.exists())
            rec = json.loads(data_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(rec["extraction_ok"])
            self.assertEqual(rec["reasoning"], "hidden chain of thought")


if __name__ == "__main__":
    unittest.main()
