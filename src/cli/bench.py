#!/usr/bin/env python3
"""Benchmark throughput for a given run profile."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from ..core.config import config_to_defaults, load_run_config
from ..core.llm_api import (
    APISettings,
    ChatRequest,
    EndpointConfig,
    OutageError,
    build_messages,
    parse_api_endpoints,
    run_chat_requests,
)
from ..core.prompts import USER_TEMPLATE, build_system_prompt
from ..core.schema_loader import load_schema
from ..utils.utils import colored, now_iso, print_header


def _apply_benchmark_defaults(defaults: dict, cfg: dict) -> dict:
    bench = cfg.get("benchmark") or {}
    out = dict(defaults)

    out["max_notes"] = bench.get("max_notes", 500)

    if "batch_size" in bench:
        out["batch_size"] = bench.get("batch_size")
    if "queue_size" in bench:
        out["queue_size"] = bench.get("queue_size")
    if "max_new_tokens" in bench:
        out["max_new_tokens"] = bench.get("max_new_tokens")

    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    base_defaults = {
        "dataset": "ncbi/Open-Patients",
        "split": "train",
        "model": None,
        "prompt_mode": "chat",
        "schema": "configs/schemas/schema.json",
        "replicas": 1,
        "batch_size": 32,
        "queue_size": 0,
        "max_notes": 500,
        "max_new_tokens": 700,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "structured_output": False,
        "schema_in_prompt": False,
        "json_out": None,
        "num_shards": 1,
        "shard_idx": 0,
        "run_tag": None,
        "api_timeout_s": 120.0,
        "api_max_retries": 4,
        "api_retry_backoff_initial_s": 1.0,
        "api_retry_backoff_max_s": 30.0,
        "api_outage_abort_after_s": 900.0,
        "api_endpoints": None,
        "api_base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "endpoint_concurrency": 8,
        "endpoint_structured_mode": "json_schema",
        "endpoint_extra_body": None,
    }

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default=None, help="Run profile YAML (configs/runs/*.yaml)"
    )
    cfg_args, remaining = config_parser.parse_known_args(argv)

    cfg = load_run_config(cfg_args.config) if cfg_args.config else {}
    defaults = dict(base_defaults)
    defaults.update(config_to_defaults(cfg))
    defaults = _apply_benchmark_defaults(defaults, cfg)
    defaults["config"] = cfg_args.config
    defaults = {k: v for k, v in defaults.items() if (k in base_defaults or k == "config")}

    ap = argparse.ArgumentParser(description="Benchmark Open-Patients enrichment throughput.")

    ap.add_argument("--config", help="Run profile YAML (configs/runs/*.yaml)")
    ap.add_argument("--dataset", help="HF dataset name")
    ap.add_argument("--split", help="dataset split (Open-Patients uses train)")
    ap.add_argument("--model", help="Fallback model name for endpoint configs")
    ap.add_argument(
        "--prompt_mode",
        choices=["chat", "plain"],
        help="Prompt formatting mode for Chat Completions messages",
    )

    ap.add_argument("--schema", help="Path to JSON schema wrapper")
    ap.add_argument(
        "--replicas",
        type=int,
        help=(
            "If >1, orchestrate replica-sharded benchmark workers in this command "
            "(equivalent to using op-bench-replicas)"
        ),
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        help="Deprecated alias for queue size (used if --queue_size is unset)",
    )
    ap.add_argument("--queue_size", type=int, help="Async request queue size (0 = auto)")
    ap.add_argument("--max_notes", type=int, help="Number of notes to benchmark (0 = all)")
    ap.add_argument("--max_new_tokens", type=int)

    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top_p", type=float)
    ap.add_argument("--seed", type=int)

    ap.add_argument("--num_shards", type=int)
    ap.add_argument("--shard_idx", type=int)
    ap.add_argument("--run_tag", help="Optional tag for multi-process benchmarks")

    ap.add_argument("--structured_output", action="store_true")
    ap.add_argument(
        "--schema_in_prompt",
        action="store_true",
        help="Embed full JSON schema in the prompt and disable endpoint structured output",
    )

    ap.add_argument(
        "--json_out",
        help="Write metrics to JSON file (defaults to benchmarks/bench_*.json)",
    )

    # API settings
    ap.add_argument("--api_timeout_s", type=float, help="Per-request timeout in seconds")
    ap.add_argument("--api_max_retries", type=int, help="Max retries per request")
    ap.add_argument("--api_retry_backoff_initial_s", type=float, help="Initial retry backoff")
    ap.add_argument("--api_retry_backoff_max_s", type=float, help="Max retry backoff")
    ap.add_argument(
        "--api_outage_abort_after_s",
        type=float,
        help="Abort benchmark if all endpoints stay unhealthy this long",
    )
    ap.add_argument(
        "--api_endpoints",
        help=(
            "JSON list of endpoint objects. Each endpoint supports: "
            "name, base_url, model, api_key_env, concurrency, structured_mode, extra_body"
        ),
    )
    ap.add_argument("--api_base_url", help="Fallback API base URL for single-endpoint mode")
    ap.add_argument("--api_key_env", help="Fallback API key env var for endpoints")
    ap.add_argument("--endpoint_concurrency", type=int, help="Fallback endpoint concurrency")
    ap.add_argument(
        "--endpoint_structured_mode",
        choices=["json_schema", "json_object", "none"],
        help="Fallback endpoint structured mode",
    )
    ap.add_argument("--endpoint_extra_body", help="Fallback endpoint extra_body JSON dict")

    ap.set_defaults(**defaults)
    return ap.parse_args(remaining)


def _strip_cli_flags(
    argv: List[str],
    *,
    flags_with_values: List[str],
    flags_without_values: List[str],
) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        matched = False
        for flag in flags_with_values:
            if tok == flag:
                i += 2
                matched = True
                break
            if tok.startswith(flag + "="):
                i += 1
                matched = True
                break
        if matched:
            continue
        if tok in flags_without_values:
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _delegate_to_bench_replicas(args: argparse.Namespace) -> None:
    if not args.config:
        raise SystemExit("--replicas requires --config so child benchmarks can be launched consistently.")
    if args.replicas < 1:
        raise SystemExit("--replicas must be >= 1")
    if args.num_shards != 1 or args.shard_idx != 0:
        raise SystemExit("Do not combine --replicas with --num_shards/--shard_idx.")
    if args.run_tag:
        raise SystemExit("Do not combine --replicas with --run_tag.")
    if args.json_out:
        raise SystemExit("Do not combine --replicas with --json_out. Use bench_metadata.json output.")

    root = Path(__file__).resolve().parents[2]
    extras = _strip_cli_flags(
        sys.argv[1:],
        flags_with_values=["--config", "--replicas"],
        flags_without_values=[],
    )
    cmd = [
        sys.executable,
        "-m",
        "src.cli.bench_replicas",
        "--config",
        args.config,
        "--replicas",
        str(args.replicas),
    ]
    if extras:
        cmd += ["--"] + extras

    print(
        colored(
            f"[bench-replicas] delegating to launcher: {' '.join(cmd)}",
            "CYAN",
        )
    )
    raise SystemExit(subprocess.call(cmd, cwd=root))


def _parse_endpoint_extra_body(raw: Optional[str]) -> Dict[str, object]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"Invalid --endpoint_extra_body JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--endpoint_extra_body must decode to a JSON object.")
    return parsed


def _endpoint_summaries(endpoints: List[EndpointConfig]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for ep in endpoints:
        out.append(
            {
                "name": ep.name,
                "base_url": ep.base_url,
                "model": ep.model,
                "api_key_env": ep.api_key_env,
                "concurrency": ep.concurrency,
                "structured_mode": ep.structured_mode,
                "extra_body": ep.extra_body,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    print_header("Open-Patients Benchmark")
    if args.config:
        print(f"Config: {colored(args.config, 'CYAN')}")
    if args.replicas > 1:
        _delegate_to_bench_replicas(args)

    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise SystemExit("--shard_idx must satisfy 0 <= shard_idx < num_shards")

    if args.json_out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        args.json_out = str(Path("benchmarks") / f"bench_{ts}_{suffix}.json")

    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = Path.cwd() / schema_path
    if not schema_path.exists():
        raise SystemExit(f"Schema not found: {schema_path}")

    schema_bundle = load_schema(schema_path)

    use_structured_output = bool(args.structured_output)
    if args.schema_in_prompt:
        if use_structured_output:
            print(
                colored(
                    "[info] --schema_in_prompt enabled; disabling --structured_output.",
                    "YELLOW",
                )
            )
        use_structured_output = False

    endpoint_extra_body = _parse_endpoint_extra_body(args.endpoint_extra_body)
    try:
        endpoints = parse_api_endpoints(
            args.api_endpoints,
            default_model=args.model,
            default_base_url=args.api_base_url,
            default_api_key_env=args.api_key_env,
            default_concurrency=args.endpoint_concurrency,
            default_structured_mode=args.endpoint_structured_mode,
            default_extra_body=endpoint_extra_body,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not endpoints:
        raise SystemExit(
            "No API endpoints configured. Set api.endpoints in --config or provide "
            "--api_endpoints / --api_base_url + --model."
        )

    print("Endpoints:")
    for ep in endpoints:
        print(
            f"  - {colored(ep.name, 'CYAN')} model={colored(ep.model, 'GREEN')} "
            f"base={colored(ep.base_url, 'WHITE')} concurrency={colored(str(ep.concurrency), 'GREEN')}"
        )

    api_settings = APISettings(
        timeout_s=float(args.api_timeout_s),
        max_retries=int(args.api_max_retries),
        retry_backoff_initial_s=float(args.api_retry_backoff_initial_s),
        retry_backoff_max_s=float(args.api_retry_backoff_max_s),
        outage_abort_after_s=float(args.api_outage_abort_after_s),
    )
    sampling_cfg = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed if args.seed != 0 else None,
    )

    system_prompt = build_system_prompt(
        schema_bundle,
        include_json_schema=bool(args.schema_in_prompt),
    )
    keys_str = json.dumps(schema_bundle.schema_keys, ensure_ascii=False)

    submitted = 0
    completed = 0
    failed = 0
    input_tokens = 0
    output_tokens = 0

    queue_size = args.queue_size if args.queue_size > 0 else args.batch_size
    if queue_size <= 0:
        queue_size = None

    pbar = tqdm(total=args.max_notes if args.max_notes else None, desc="bench")

    def iter_requests():
        nonlocal submitted
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        for row in ds:
            if args.max_notes and submitted >= args.max_notes:
                break

            row_id = row.get("_id") or row.get("id")
            if args.num_shards > 1:
                if not row_id:
                    continue
                h = int(hashlib.md5(str(row_id).encode("utf-8")).hexdigest(), 16)
                if (h % args.num_shards) != args.shard_idx:
                    continue

            note = row.get("description", "")
            if not isinstance(note, str):
                continue

            user = USER_TEMPLATE.format(note=note, keys=keys_str)
            messages = build_messages(args.prompt_mode, system_prompt, user)
            submitted += 1
            pbar.update(1)
            yield ChatRequest(request_id=str(row_id or submitted), messages=messages, metadata={})

    async def on_result(result) -> None:
        nonlocal completed, failed, input_tokens, output_tokens
        completed += 1
        usage = result.usage or {}
        input_tokens += int(usage.get("prompt_tokens", 0) or 0)
        output_tokens += int(usage.get("completion_tokens", 0) or 0)
        if result.error:
            failed += 1

    start_iso = now_iso()
    total_start = time.perf_counter()

    scheduler_stats: Dict[str, object] = {}
    aborted_reason: Optional[str] = None
    gen_t0 = time.perf_counter()
    try:
        _, scheduler_stats = asyncio.run(
            run_chat_requests(
                requests=iter_requests(),
                endpoints=endpoints,
                api_settings=api_settings,
                sampling_cfg=sampling_cfg,
                structured_output=use_structured_output,
                json_schema=(schema_bundle.wrapper if use_structured_output else None),
                on_result=on_result,
                queue_size=queue_size,
            )
        )
    except OutageError as exc:
        aborted_reason = str(exc)
        scheduler_stats = dict(exc.stats or {})
    finally:
        gen_time = time.perf_counter() - gen_t0
        pbar.close()

    total_time = time.perf_counter() - total_start
    end_iso = now_iso()

    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    metrics = {
        "config": args.config,
        "run_tag": args.run_tag,
        "max_notes": args.max_notes,
        "queue_size": queue_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_mode": args.prompt_mode,
        "structured_output": bool(use_structured_output),
        "schema_in_prompt": bool(args.schema_in_prompt),
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "start_time": start_iso,
        "end_time": end_iso,
        "notes": submitted,
        "completed": completed,
        "failed_requests": failed,
        "batches": completed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "total_time_s": total_time,
        "gen_time_s": gen_time,
        "notes_per_s": _safe_div(submitted, total_time),
        "input_toks_per_s": _safe_div(input_tokens, gen_time),
        "output_toks_per_s": _safe_div(output_tokens, gen_time),
        "total_toks_per_s": _safe_div(input_tokens + output_tokens, gen_time),
        "avg_input_toks_per_note": _safe_div(input_tokens, submitted),
        "avg_output_toks_per_note": _safe_div(output_tokens, submitted),
        "api": {
            "timeout_s": api_settings.timeout_s,
            "max_retries": api_settings.max_retries,
            "retry_backoff_initial_s": api_settings.retry_backoff_initial_s,
            "retry_backoff_max_s": api_settings.retry_backoff_max_s,
            "outage_abort_after_s": api_settings.outage_abort_after_s,
            "endpoints": _endpoint_summaries(endpoints),
        },
        "scheduler": scheduler_stats,
        "aborted": aborted_reason is not None,
        "aborted_reason": aborted_reason,
    }

    print("\n" + colored("Benchmark results", "CYAN"))
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {colored(k, 'WHITE')}: {colored(f'{v:.3f}', 'GREEN')}")
        else:
            print(f"  {colored(k, 'WHITE')}: {colored(str(v), 'GREEN')}")

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute() and out_path.parent == Path("."):
            out_path = Path("benchmarks") / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nWrote metrics: {colored(str(out_path.resolve()), 'CYAN')}")

    if aborted_reason is not None:
        raise SystemExit(f"Benchmark aborted: {aborted_reason}")


if __name__ == "__main__":
    main()
