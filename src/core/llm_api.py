"""
OpenAI-compatible API inference helpers (Chat Completions + dynamic endpoint queue).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import is validated by runtime usage/tests
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AsyncOpenAI,
        RateLimitError,
    )
except Exception:  # pragma: no cover
    APIConnectionError = tuple()  # type: ignore[assignment]
    APIStatusError = tuple()  # type: ignore[assignment]
    APITimeoutError = tuple()  # type: ignore[assignment]
    RateLimitError = tuple()  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment]


_STRUCTURED_MODES = {"json_schema", "json_object", "none"}


@dataclass
class APISettings:
    timeout_s: float = 120.0
    max_retries: int = 4
    retry_backoff_initial_s: float = 1.0
    retry_backoff_max_s: float = 30.0
    outage_abort_after_s: float = 900.0


@dataclass
class EndpointConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    concurrency: int = 8
    structured_mode: str = "json_schema"
    extra_body: Dict[str, Any] = field(default_factory=dict)
    serve: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatRequest:
    request_id: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResult:
    request_id: str
    endpoint_name: str
    text: str
    reasoning: Optional[str]
    usage: Dict[str, int]
    attempts: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OutageError(RuntimeError):
    def __init__(self, message: str, stats: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.stats = stats or {}


ClientFactory = Callable[[EndpointConfig, float], Any]
ResultCallback = Callable[[ChatResult], Awaitable[None]]


def parse_api_endpoints(
    raw: Any,
    *,
    default_model: Optional[str] = None,
    default_base_url: Optional[str] = None,
    default_api_key_env: str = "OPENAI_API_KEY",
    default_concurrency: int = 8,
    default_structured_mode: str = "json_schema",
    default_extra_body: Optional[Dict[str, Any]] = None,
) -> List[EndpointConfig]:
    """
    Parse endpoint config from YAML/JSON CLI values.
    """
    endpoints_raw = raw
    if isinstance(raw, str):
        try:
            endpoints_raw = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"Invalid --api_endpoints JSON: {exc}") from exc

    if endpoints_raw is None:
        if default_base_url and default_model:
            endpoints_raw = [
                {
                    "name": "default",
                    "base_url": default_base_url,
                    "model": default_model,
                    "api_key_env": default_api_key_env,
                    "concurrency": default_concurrency,
                    "structured_mode": default_structured_mode,
                    "extra_body": default_extra_body or {},
                }
            ]
        else:
            endpoints_raw = []

    if not isinstance(endpoints_raw, list):
        raise ValueError("api.endpoints must be a list (or JSON list).")

    out: List[EndpointConfig] = []
    seen_names = set()
    for idx, item in enumerate(endpoints_raw):
        if not isinstance(item, dict):
            raise ValueError(f"api.endpoints[{idx}] must be a mapping.")

        name = str(item.get("name") or f"ep{idx}")
        if name in seen_names:
            raise ValueError(f"Duplicate endpoint name: {name}")
        seen_names.add(name)

        base_url = item.get("base_url") or default_base_url
        model = item.get("model") or default_model
        if not base_url or not model:
            raise ValueError(
                f"api.endpoints[{idx}] requires base_url and model "
                "(or provide global defaults)."
            )
        api_key_env = str(item.get("api_key_env") or default_api_key_env)
        try:
            concurrency = int(item.get("concurrency") or default_concurrency)
        except Exception as exc:
            raise ValueError(f"Invalid concurrency for endpoint {name}") from exc
        if concurrency < 1:
            raise ValueError(f"Endpoint {name} has invalid concurrency={concurrency}")

        structured_mode = str(item.get("structured_mode") or default_structured_mode).lower()
        if structured_mode not in _STRUCTURED_MODES:
            raise ValueError(
                f"Endpoint {name} has invalid structured_mode={structured_mode}; "
                f"expected one of {sorted(_STRUCTURED_MODES)}"
            )

        extra_body = item.get("extra_body")
        if extra_body is None:
            extra_body = default_extra_body or {}
        if isinstance(extra_body, str):
            try:
                extra_body = json.loads(extra_body)
            except Exception as exc:
                raise ValueError(f"Invalid extra_body JSON for endpoint {name}: {exc}") from exc
        if not isinstance(extra_body, dict):
            raise ValueError(f"extra_body for endpoint {name} must be a mapping.")

        serve = item.get("serve") or {}
        if not isinstance(serve, dict):
            raise ValueError(f"serve config for endpoint {name} must be a mapping.")

        out.append(
            EndpointConfig(
                name=name,
                base_url=str(base_url),
                model=str(model),
                api_key_env=api_key_env,
                concurrency=concurrency,
                structured_mode=structured_mode,
                extra_body=dict(extra_body),
                serve=dict(serve),
            )
        )
    return out


def build_messages(prompt_mode: str, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    if prompt_mode == "plain":
        return [
            {
                "role": "user",
                "content": f"{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n",
            }
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_response_format(
    structured_mode: str,
    json_schema: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if structured_mode == "none":
        return None
    if structured_mode == "json_object":
        return {"type": "json_object"}
    if structured_mode == "json_schema":
        if not json_schema:
            return None
        inner = json_schema
        if "json_schema" in json_schema and isinstance(json_schema["json_schema"], dict):
            inner = json_schema["json_schema"].get("schema") or json_schema
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "open_patients",
                "schema": inner,
            },
        }
    return None


def _default_client_factory(endpoint: EndpointConfig, timeout_s: float):
    if AsyncOpenAI is None:
        raise RuntimeError(
            "openai package is not installed. Install dependencies with `uv sync`."
        )
    api_key = os.environ.get(endpoint.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key for endpoint '{endpoint.name}'. "
            f"Set environment variable {endpoint.api_key_env}."
        )
    return AsyncOpenAI(base_url=endpoint.base_url, api_key=api_key, timeout=timeout_s)


def _extract_text_from_choice_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                elif item.get("type") in {"output_text", "text"} and isinstance(
                    item.get("content"), str
                ):
                    chunks.append(item["content"])
            else:
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    chunks.append(text_attr)
        return "".join(chunks)
    return str(content)


def _normalize_reasoning(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        out = value.strip()
        return out or None
    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    chunks.append(s)
                continue
            if isinstance(item, dict):
                for key in ("text", "reasoning", "reasoning_content", "summary", "content"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        chunks.append(v.strip())
                        break
                continue
            s = str(item).strip()
            if s:
                chunks.append(s)
        if not chunks:
            return None
        return "\n\n".join(chunks)
    if isinstance(value, dict):
        for key in ("text", "reasoning", "reasoning_content", "summary", "content"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        dumped = json.dumps(value, ensure_ascii=False).strip()
        return dumped or None
    out = str(value).strip()
    return out or None


def _extract_reasoning_from_message(message: Any) -> Optional[str]:
    if message is None:
        return None

    # Common provider field (OpenRouter, some OpenAI-compatible servers).
    reasoning = _normalize_reasoning(getattr(message, "reasoning", None))
    if reasoning:
        return reasoning

    # Common field from reasoning parsers (e.g. some vLLM-compatible servers).
    reasoning = _normalize_reasoning(getattr(message, "reasoning_content", None))
    if reasoning:
        return reasoning

    reasoning = _normalize_reasoning(getattr(message, "reasoning_details", None))
    if reasoning:
        return reasoning

    # Fallback to dump-based fields for SDK/client variants.
    dump_fn = getattr(message, "model_dump", None)
    if callable(dump_fn):
        try:
            dumped = dump_fn()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            reasoning = _normalize_reasoning(dumped.get("reasoning"))
            if reasoning:
                return reasoning
            reasoning = _normalize_reasoning(dumped.get("reasoning_content"))
            if reasoning:
                return reasoning
            reasoning = _normalize_reasoning(dumped.get("reasoning_details"))
            if reasoning:
                return reasoning

    return None


def _usage_dict(usage_obj: Any) -> Dict[str, int]:
    if usage_obj is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage_obj, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _status_code_from_exc(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _is_retryable_error(exc: Exception) -> bool:
    if APITimeoutError and isinstance(exc, APITimeoutError):
        return True
    if APIConnectionError and isinstance(exc, APIConnectionError):
        return True
    if RateLimitError and isinstance(exc, RateLimitError):
        return True
    if APIStatusError and isinstance(exc, APIStatusError):
        code = _status_code_from_exc(exc)
        return bool(code == 429 or (code is not None and code >= 500))
    code = _status_code_from_exc(exc)
    if code is not None:
        return code == 429 or code >= 500
    # Conservative fallback for client-less tests/mocks.
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


async def request_chat_completion(
    *,
    client: Any,
    endpoint: EndpointConfig,
    api_settings: APISettings,
    request: ChatRequest,
    sampling_cfg: Dict[str, Any],
    structured_output: bool,
    json_schema: Optional[Dict[str, Any]],
) -> ChatResult:
    structured_mode = endpoint.structured_mode if structured_output else "none"
    response_format = _build_response_format(structured_mode, json_schema)

    request_kwargs: Dict[str, Any] = {
        "model": endpoint.model,
        "messages": request.messages,
        "temperature": sampling_cfg.get("temperature"),
        "top_p": sampling_cfg.get("top_p"),
        "max_tokens": sampling_cfg.get("max_new_tokens"),
    }
    seed = sampling_cfg.get("seed")
    if seed is not None:
        request_kwargs["seed"] = seed
    if response_format is not None:
        request_kwargs["response_format"] = response_format
    if endpoint.extra_body:
        request_kwargs["extra_body"] = endpoint.extra_body

    max_attempts = max(1, int(api_settings.max_retries) + 1)
    delay = max(float(api_settings.retry_backoff_initial_s), 0.0)
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.chat.completions.create(**request_kwargs)
            choices = getattr(resp, "choices", []) or []
            text = ""
            if choices:
                message = getattr(choices[0], "message", None)
                text = _extract_text_from_choice_content(getattr(message, "content", ""))
                reasoning = _extract_reasoning_from_message(message)
            else:
                reasoning = None
            return ChatResult(
                request_id=request.request_id,
                endpoint_name=endpoint.name,
                text=text,
                reasoning=reasoning,
                usage=_usage_dict(getattr(resp, "usage", None)),
                attempts=attempt,
                error=None,
                metadata=dict(request.metadata),
            )
        except Exception as exc:  # pragma: no cover - exercised via mock-based tests
            err = f"{type(exc).__name__}: {exc}"
            if attempt >= max_attempts or not _is_retryable_error(exc):
                return ChatResult(
                    request_id=request.request_id,
                    endpoint_name=endpoint.name,
                    text="",
                    reasoning=None,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    attempts=attempt,
                    error=err,
                    metadata=dict(request.metadata),
                )
            await asyncio.sleep(min(delay, float(api_settings.retry_backoff_max_s)))
            delay = min(delay * 2 if delay > 0 else 1.0, float(api_settings.retry_backoff_max_s))

    return ChatResult(
        request_id=request.request_id,
        endpoint_name=endpoint.name,
        text="",
        reasoning=None,
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        attempts=max_attempts,
        error="Unknown failure",
        metadata=dict(request.metadata),
    )


async def run_chat_requests(
    *,
    requests: Iterable[ChatRequest],
    endpoints: Sequence[EndpointConfig],
    api_settings: APISettings,
    sampling_cfg: Dict[str, Any],
    structured_output: bool,
    json_schema: Optional[Dict[str, Any]],
    on_result: Optional[ResultCallback] = None,
    client_factory: Optional[ClientFactory] = None,
    queue_size: Optional[int] = None,
) -> Tuple[List[ChatResult], Dict[str, Any]]:
    if not endpoints:
        raise ValueError("At least one API endpoint is required.")

    total_workers = sum(max(1, ep.concurrency) for ep in endpoints)
    queue_cap = queue_size or max(2, total_workers * 2)
    req_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_cap)
    res_queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    stop_event = asyncio.Event()
    state_lock = asyncio.Lock()

    make_client = client_factory or _default_client_factory
    results: List[ChatResult] = []

    endpoint_stats: Dict[str, Dict[str, int]] = {
        ep.name: {"submitted": 0, "succeeded": 0, "failed": 0} for ep in endpoints
    }
    submitted = 0
    completed = 0
    producer_done = False
    start_monotonic = time.monotonic()
    last_success = start_monotonic
    endpoint_last_success = {ep.name: start_monotonic for ep in endpoints}
    endpoint_last_error = {ep.name: 0.0 for ep in endpoints}
    outage_reason: Optional[str] = None

    async def producer() -> None:
        nonlocal submitted, producer_done
        try:
            for req in requests:
                if stop_event.is_set():
                    break
                while True:
                    if stop_event.is_set():
                        break
                    try:
                        await asyncio.wait_for(req_queue.put(req), timeout=0.25)
                        submitted += 1
                        break
                    except asyncio.TimeoutError:
                        continue
                if stop_event.is_set():
                    break
        finally:
            producer_done = True
            for _ in range(total_workers):
                await req_queue.put(sentinel)

    async def worker(endpoint: EndpointConfig) -> None:
        nonlocal last_success
        client = make_client(endpoint, float(api_settings.timeout_s))
        while True:
            item = await req_queue.get()
            if item is sentinel:
                req_queue.task_done()
                break
            req: ChatRequest = item
            result: ChatResult
            try:
                async with state_lock:
                    endpoint_stats[endpoint.name]["submitted"] += 1
                result = await request_chat_completion(
                    client=client,
                    endpoint=endpoint,
                    api_settings=api_settings,
                    request=req,
                    sampling_cfg=sampling_cfg,
                    structured_output=structured_output,
                    json_schema=json_schema,
                )
            except Exception as exc:  # pragma: no cover
                result = ChatResult(
                    request_id=req.request_id,
                    endpoint_name=endpoint.name,
                    text="",
                    reasoning=None,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    attempts=1,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=dict(req.metadata),
                )

            now = time.monotonic()
            async with state_lock:
                if result.error:
                    endpoint_stats[endpoint.name]["failed"] += 1
                    endpoint_last_error[endpoint.name] = now
                else:
                    endpoint_stats[endpoint.name]["succeeded"] += 1
                    endpoint_last_success[endpoint.name] = now
                    last_success = now
            await res_queue.put(result)
            req_queue.task_done()
        await res_queue.put(sentinel)

    async def collector() -> None:
        nonlocal completed
        done_workers = 0
        while done_workers < total_workers:
            item = await res_queue.get()
            if item is sentinel:
                done_workers += 1
                continue
            result: ChatResult = item
            completed += 1
            if on_result is not None:
                await on_result(result)
            else:
                results.append(result)

    async def outage_monitor() -> None:
        nonlocal outage_reason
        while True:
            await asyncio.sleep(1.0)
            if stop_event.is_set():
                return
            if producer_done and completed >= submitted and req_queue.empty():
                return
            if submitted == 0:
                continue
            async with state_lock:
                all_unhealthy = all(
                    endpoint_last_error[name] > endpoint_last_success[name]
                    for name in endpoint_last_success
                )
                elapsed = time.monotonic() - last_success
            if all_unhealthy and elapsed >= float(api_settings.outage_abort_after_s):
                outage_reason = (
                    f"all endpoints unhealthy for {elapsed:.1f}s "
                    f"(threshold={api_settings.outage_abort_after_s}s)"
                )
                stop_event.set()
                return

    producer_task = asyncio.create_task(producer())
    worker_tasks = [
        asyncio.create_task(worker(ep))
        for ep in endpoints
        for _ in range(max(1, ep.concurrency))
    ]
    collector_task = asyncio.create_task(collector())
    monitor_task = asyncio.create_task(outage_monitor())

    try:
        await producer_task
        await asyncio.gather(*worker_tasks)
        await collector_task
    finally:
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task

    stats = {
        "submitted": submitted,
        "completed": completed,
        "outage_abort": outage_reason is not None,
        "outage_reason": outage_reason,
        "endpoint_stats": endpoint_stats,
        "queue_size": queue_cap,
    }
    if outage_reason is not None:
        raise OutageError(outage_reason, stats)

    return results, stats
