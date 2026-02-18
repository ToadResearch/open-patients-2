#!/usr/bin/env python3
"""
Main entry point for the Open-Patients enrichment pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from ..core.config import config_to_defaults, load_run_config
from ..core.extraction import derive_source_url, ensure_schema, load_usmle_id_to_row
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
from ..core.writer import JSONLShardedWriter, ProcessedIdWriter, load_processed_ids
from ..utils.utils import (
    colored,
    now_iso,
    print_header,
    safe_json_extract,
    split_reasoning_and_final,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    base_defaults = {
        "dataset": "ncbi/Open-Patients",
        "split": "train",
        "model": None,
        "prompt_mode": "chat",
        "out_dir": None,
        "processed_ids": "processed_ids.txt",
        "failed_ids": "failed_ids.txt",
        "schema": "configs/schemas/schema.json",
        "usmle_mapping": "configs/usmle_mapping.json",
        "run_id": None,
        "run_tag": None,
        "replicas": 1,
        "queue_size": 0,
        "max_notes": 0,
        "max_new_tokens": 700,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "shard_size": 50_000,
        "resume": False,
        "structured_output": False,
        "schema_in_prompt": False,
        "combine_shards": True,
        "num_shards": 1,
        "shard_idx": 0,
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
    defaults["config"] = cfg_args.config
    defaults = {k: v for k, v in defaults.items() if (k in base_defaults or k == "config")}

    ap = argparse.ArgumentParser(
        description="Enrich Open-Patients dataset with structured clinical fields."
    )
    ap.add_argument("--config", help="Run profile YAML (configs/runs/*.yaml)")
    ap.add_argument("--dataset", help="HF dataset name")
    ap.add_argument("--split", help="dataset split (Open-Patients uses train)")
    ap.add_argument(
        "--model",
        help="Fallback model name for endpoints missing model (or single-endpoint mode)",
    )
    ap.add_argument(
        "--prompt_mode",
        choices=["chat", "plain"],
        help="Prompt formatting mode for Chat Completions messages",
    )

    ap.add_argument("--out_dir", help="Output directory for enriched JSONL shards")
    ap.add_argument(
        "--processed_ids",
        help="Resume marker file (in out_dir by default)",
    )
    ap.add_argument(
        "--failed_ids",
        help="Failed-id output file (in out_dir by default)",
    )
    ap.add_argument(
        "--schema",
        help="Path to JSON schema wrapper (default: configs/schemas/schema.json)",
    )
    ap.add_argument(
        "--usmle_mapping",
        help="Path to configs/usmle_mapping.json (for usmle-<num> -> HF viewer row mapping)",
    )
    ap.add_argument(
        "--run_id",
        help="Optional run id subfolder name (under out_dir)",
    )
    ap.add_argument(
        "--run_tag",
        help="Optional tag to prefix output shards/metadata (for multi-process runs)",
    )
    ap.add_argument(
        "--replicas",
        type=int,
        help=(
            "If >1, orchestrate replica-sharded workers in this command "
            "(equivalent to using op-replicas)"
        ),
    )

    ap.add_argument(
        "--queue_size",
        type=int,
        help="Async request queue size (0 = auto based on endpoint concurrency)",
    )
    ap.add_argument("--max_notes", type=int, help="0 = all")
    ap.add_argument("--max_new_tokens", type=int)

    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top_p", type=float)
    ap.add_argument("--seed", type=int)

    ap.add_argument("--shard_size", type=int, help="JSONL records per output shard")
    ap.add_argument("--resume", action="store_true", help="Skip IDs already in processed_ids file")

    ap.add_argument(
        "--structured_output",
        action="store_true",
        help="Enable endpoint-structured JSON response mode where supported",
    )
    ap.add_argument(
        "--schema_in_prompt",
        action="store_true",
        help="Embed full JSON schema in the prompt and disable endpoint structured output",
    )
    ap.add_argument(
        "--no_combine_shards",
        dest="combine_shards",
        action="store_false",
        help="Do not concatenate shard files into a final data.jsonl at end of run.",
    )

    # Manual dataset sharding across processes
    ap.add_argument("--num_shards", type=int)
    ap.add_argument("--shard_idx", type=int)

    # API settings
    ap.add_argument("--api_timeout_s", type=float, help="Per-request timeout in seconds")
    ap.add_argument("--api_max_retries", type=int, help="Max retries per request")
    ap.add_argument(
        "--api_retry_backoff_initial_s",
        type=float,
        help="Initial retry backoff seconds",
    )
    ap.add_argument(
        "--api_retry_backoff_max_s",
        type=float,
        help="Max retry backoff seconds",
    )
    ap.add_argument(
        "--api_outage_abort_after_s",
        type=float,
        help="Abort run if all endpoints remain unhealthy for this many seconds",
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
    ap.add_argument(
        "--endpoint_concurrency",
        type=int,
        help="Fallback per-endpoint concurrency if not set in endpoint config",
    )
    ap.add_argument(
        "--endpoint_structured_mode",
        choices=["json_schema", "json_object", "none"],
        help="Fallback per-endpoint structured mode",
    )
    ap.add_argument(
        "--endpoint_extra_body",
        help="Fallback JSON dict merged as extra_body for endpoints missing extra_body",
    )

    ap.set_defaults(**defaults)
    return ap.parse_args(remaining)


def _combine_jsonl_shards(shards_dir: Path, shard_prefix: str, out_path: Path) -> int:
    """
    Concatenate shard files from shards_dir into out_path.

    Returns the number of shard files combined.
    """
    pattern = f"{shard_prefix}_[0-9][0-9][0-9][0-9][0-9].jsonl"
    shard_paths = sorted(shards_dir.glob(pattern))
    if not shard_paths:
        return 0

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as out_f:
        for p in shard_paths:
            with p.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f)
    tmp_path.replace(out_path)
    return len(shard_paths)


def _resolve_tracking_path(
    raw_path: str,
    out_dir: Path,
    run_tag: Optional[str],
) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if run_tag:
        return out_dir / f"{p.stem}_{run_tag}{p.suffix}"
    return out_dir / p


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


def _delegate_to_replicas(args: argparse.Namespace) -> None:
    if not args.config:
        raise SystemExit("--replicas requires --config so child workers can be launched consistently.")
    if args.replicas < 1:
        raise SystemExit("--replicas must be >= 1")
    if args.num_shards != 1 or args.shard_idx != 0:
        raise SystemExit("Do not combine --replicas with --num_shards/--shard_idx.")
    if args.run_tag:
        raise SystemExit("Do not combine --replicas with --run_tag.")

    root = Path(__file__).resolve().parents[2]
    extras = _strip_cli_flags(
        sys.argv[1:],
        flags_with_values=["--config", "--replicas", "--run_id"],
        flags_without_values=[],
    )
    cmd = [
        sys.executable,
        "-m",
        "src.cli.launch",
        "--config",
        args.config,
        "--replicas",
        str(args.replicas),
    ]
    if args.run_id:
        cmd += ["--run_id", args.run_id]
    if extras:
        cmd += ["--"] + extras

    print(
        colored(
            f"[replicas] delegating to launcher: {' '.join(cmd)}",
            "CYAN",
        )
    )
    raise SystemExit(subprocess.call(cmd, cwd=root))


def main() -> None:
    args = parse_args()
    print_header("Open-Patients Worker")
    if args.config:
        print(f"Config: {colored(args.config, 'CYAN')}")
    if args.replicas > 1:
        _delegate_to_replicas(args)
    if not args.out_dir:
        raise SystemExit("Missing --out_dir (or run.out_dir in --config).")

    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise SystemExit("--shard_idx must satisfy 0 <= shard_idx < num_shards")

    def _resolve_out_dir(p: Path) -> Path:
        if p.is_absolute():
            return p
        if p.parts and p.parts[0] == "outputs":
            return p
        return Path("outputs") / p

    base_out_dir = _resolve_out_dir(Path(args.out_dir))
    if base_out_dir.exists() and base_out_dir.is_file():
        raise SystemExit(f"out_dir is a file: {base_out_dir}")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id
    out_dir = base_out_dir
    if run_id:
        out_dir = base_out_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
    elif not args.resume:
        for _ in range(20):
            ts = time.strftime("%Y%m%d_%H%M%S")
            suffix = uuid.uuid4().hex[:6]
            run_id = f"run_{ts}_{suffix}"
            candidate = base_out_dir / run_id
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                out_dir = candidate
                break
            except FileExistsError:
                time.sleep(0.01)
        if out_dir == base_out_dir:
            raise SystemExit("Failed to create unique run directory under out_dir.")

    processed_path = _resolve_tracking_path(args.processed_ids, out_dir, args.run_tag)
    failed_ids_path = _resolve_tracking_path(args.failed_ids, out_dir, args.run_tag)

    processed_ids = load_processed_ids(processed_path) if args.resume else set()
    processed_writer = ProcessedIdWriter(processed_path, flush_every=2000)

    # Load USMLE row mapping (optional but recommended)
    usmle_map_path = Path(args.usmle_mapping)
    if not usmle_map_path.is_absolute():
        usmle_map_path = Path.cwd() / usmle_map_path

    usmle_id_to_row = load_usmle_id_to_row(usmle_map_path) if usmle_map_path.exists() else None

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

    sampling_cfg = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed if args.seed != 0 else None,
    )
    api_settings = APISettings(
        timeout_s=float(args.api_timeout_s),
        max_retries=int(args.api_max_retries),
        retry_backoff_initial_s=float(args.api_retry_backoff_initial_s),
        retry_backoff_max_s=float(args.api_retry_backoff_max_s),
        outage_abort_after_s=float(args.api_outage_abort_after_s),
    )
    json_schema = schema_bundle.wrapper if use_structured_output else None

    shard_prefix = f"data_shard_{args.run_tag}" if args.run_tag else "data_shard"
    failed_prefix = f"failed_records_{args.run_tag}" if args.run_tag else "failed_records"
    shards_dir = out_dir / "shards"

    # One-time layout migration: move legacy shard files from out_dir/ to out_dir/shards/.
    legacy_shards = sorted(out_dir.glob("data_shard*.jsonl"))
    if legacy_shards:
        shards_dir.mkdir(parents=True, exist_ok=True)
        for p in legacy_shards:
            dest = shards_dir / p.name
            if not dest.exists():
                p.replace(dest)

    writer = JSONLShardedWriter(
        out_dir=shards_dir,
        shard_size=args.shard_size,
        name_prefix=shard_prefix,
    )
    failed_writer: Optional[JSONLShardedWriter] = None
    failed_ids_writer: Optional[ProcessedIdWriter] = None

    keys_str = json.dumps(schema_bundle.schema_keys, ensure_ascii=False)
    system_prompt = build_system_prompt(
        schema_bundle,
        include_json_schema=bool(args.schema_in_prompt),
    )

    n_total = 0
    n_written = 0
    n_skipped = 0
    n_failed = 0
    n_submitted = 0
    input_tokens = 0
    output_tokens = 0

    start_iso = now_iso()
    start_perf = time.perf_counter()

    pbar = tqdm(total=args.max_notes if args.max_notes else None, desc="enrich")

    def _write_failure_artifacts(
        _id: str,
        endpoint_name: str,
        attempts: int,
        reason: str,
        error: Optional[str],
        raw_output: str,
    ) -> None:
        nonlocal failed_writer, failed_ids_writer
        if failed_writer is None:
            failed_writer = JSONLShardedWriter(
                out_dir=shards_dir,
                shard_size=args.shard_size,
                name_prefix=failed_prefix,
            )
        if failed_ids_writer is None:
            failed_ids_writer = ProcessedIdWriter(failed_ids_path, flush_every=2000)

        created_at = now_iso()
        failed_writer.write(
            {
                "id": _id,
                "endpoint": endpoint_name,
                "attempts": attempts,
                "reason": reason,
                "error": error,
                "raw_output": raw_output,
                "created_at": created_at,
            }
        )
        failed_ids_writer.add(_id)

    def iter_requests():
        nonlocal n_total, n_skipped, n_submitted
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        for row in ds:
            if args.max_notes and n_submitted >= args.max_notes:
                break

            _id = row.get("_id")
            note = row.get("description", "")
            if not _id or not isinstance(note, str):
                continue

            n_total += 1

            if args.num_shards > 1:
                h = int(hashlib.md5(_id.encode("utf-8")).hexdigest(), 16)
                if (h % args.num_shards) != args.shard_idx:
                    continue

            if args.resume and _id in processed_ids:
                n_skipped += 1
                continue

            user = USER_TEMPLATE.format(note=note, keys=keys_str)
            messages = build_messages(args.prompt_mode, system_prompt, user)

            n_submitted += 1
            pbar.update(1)
            yield ChatRequest(
                request_id=_id,
                messages=messages,
                metadata={"row": row},
            )

    async def on_result(result) -> None:
        nonlocal n_written, n_failed, input_tokens, output_tokens
        _id = result.request_id
        row = result.metadata.get("row", {}) if isinstance(result.metadata, dict) else {}
        source_url = derive_source_url(_id, usmle_id_to_row)
        raw_text = result.text or ""
        tag_reasoning, final_text = split_reasoning_and_final(raw_text)
        reasoning = result.reasoning
        if reasoning and tag_reasoning and tag_reasoning not in reasoning:
            reasoning = f"{reasoning}\n\n{tag_reasoning}"
        elif (not reasoning) and tag_reasoning:
            reasoning = tag_reasoning

        usage = result.usage or {}
        input_tokens += int(usage.get("prompt_tokens", 0) or 0)
        output_tokens += int(usage.get("completion_tokens", 0) or 0)

        created_at = now_iso()

        if result.error:
            n_failed += 1
            rec = {
                "id": _id,
                "patient_note": row.get("description", ""),
                "source": source_url,
                "extraction_ok": False,
                "reasoning": reasoning,
                "model_output_raw": raw_text,
                "model_error": result.error,
                "created_at": created_at,
            }
            writer.write(rec)
            processed_writer.add(_id)
            n_written += 1
            _write_failure_artifacts(
                _id=_id,
                endpoint_name=result.endpoint_name,
                attempts=result.attempts,
                reason="request_error",
                error=result.error,
                raw_output=raw_text,
            )
            return

        parsed: Optional[dict]
        if use_structured_output:
            try:
                parsed = json.loads(final_text)
            except Exception:
                parsed = safe_json_extract(final_text) or safe_json_extract(raw_text)
        else:
            parsed = safe_json_extract(final_text) or safe_json_extract(raw_text)

        if parsed is None or not isinstance(parsed, dict):
            n_failed += 1
            rec = {
                "id": _id,
                "patient_note": row.get("description", ""),
                "source": source_url,
                "extraction_ok": False,
                "reasoning": reasoning,
                "model_output_raw": raw_text,
                "created_at": created_at,
            }
            writer.write(rec)
            processed_writer.add(_id)
            n_written += 1
            _write_failure_artifacts(
                _id=_id,
                endpoint_name=result.endpoint_name,
                attempts=result.attempts,
                reason="parse_error",
                error=None,
                raw_output=raw_text,
            )
            return

        parsed = ensure_schema(parsed, schema_bundle.schema_keys, schema_bundle.typed_list_names)

        rec = {
            "id": _id,
            "patient_note": row.get("description", ""),
            "source": source_url,
            "reasoning": reasoning,
        }

        for field_name in schema_bundle.scalar_names:
            rec[field_name] = parsed[field_name]

        for field_def in schema_bundle.typed_list_fields:
            field_name = field_def["name"]
            rec[field_name] = parsed.get(field_name, [])

        rec["extraction_ok"] = True
        rec["created_at"] = created_at

        writer.write(rec)
        processed_writer.add(_id)
        n_written += 1

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
                json_schema=json_schema,
                on_result=on_result,
                queue_size=(args.queue_size if args.queue_size > 0 else None),
            )
        )
    except OutageError as exc:
        aborted_reason = str(exc)
        scheduler_stats = dict(exc.stats or {})
    finally:
        gen_time = time.perf_counter() - gen_t0
        pbar.close()
        writer.close()
        processed_writer.close()
        if failed_writer is not None:
            failed_writer.close()
        if failed_ids_writer is not None:
            failed_ids_writer.close()

    combined_path: Path | None = None
    combined_shards = 0
    if args.combine_shards and (args.run_tag is None) and (args.num_shards == 1):
        combined_path = out_dir / "data.jsonl"
        combined_shards = _combine_jsonl_shards(shards_dir, shard_prefix, combined_path)

    end_iso = now_iso()
    total_time = time.perf_counter() - start_perf

    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    run_config = load_run_config(args.config) if args.config else None
    metadata = {
        "run_id": run_id,
        "run_tag": args.run_tag,
        "base_out_dir": str(base_out_dir),
        "out_dir": str(out_dir),
        "shards_dir": str(shards_dir),
        "combined_jsonl": str(combined_path) if combined_path else None,
        "combined_shards": combined_shards,
        "resume": bool(args.resume),
        "config_path": args.config,
        "config": run_config,
        "schema_path": str(schema_path),
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_s": total_time,
        "gen_time_s": gen_time,
        "notes_seen": n_total,
        "notes_submitted": n_submitted,
        "notes_written": n_written,
        "notes_skipped": n_skipped,
        "notes_failed": n_failed,
        "structured_output": bool(use_structured_output),
        "schema_in_prompt": bool(args.schema_in_prompt),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "notes_per_s": _safe_div(n_submitted, total_time),
        "input_toks_per_s": _safe_div(input_tokens, gen_time),
        "output_toks_per_s": _safe_div(output_tokens, gen_time),
        "total_toks_per_s": _safe_div(input_tokens + output_tokens, gen_time),
        "avg_input_toks_per_note": _safe_div(input_tokens, n_submitted),
        "avg_output_toks_per_note": _safe_div(output_tokens, n_submitted),
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
        "args": vars(args),
    }
    metadata_name = f"run_metadata_{args.run_tag}.json" if args.run_tag else "run_metadata.json"
    metadata_path = out_dir / metadata_name
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + colored("Done.", "GREEN" if aborted_reason is None else "YELLOW"))
    print(f"  seen:      {colored(str(n_total), 'GREEN')}")
    print(f"  submitted: {colored(str(n_submitted), 'GREEN')}")
    print(f"  wrote:     {colored(str(n_written), 'GREEN')}")
    print(f"  skipped:   {colored(str(n_skipped), 'YELLOW')}")
    failed_color = "RED" if n_failed else "GREEN"
    print(f"  failed:    {colored(str(n_failed), failed_color)}")
    print(f"  out_dir:   {colored(str(out_dir.resolve()), 'CYAN')}")
    print(f"  shards:    {colored(str(shards_dir.resolve()), 'CYAN')}")
    if combined_path:
        print(f"  data:      {colored(str(combined_path.resolve()), 'CYAN')}")
    if failed_ids_writer is not None:
        print(f"  failed_ids:{colored(str(failed_ids_path.resolve()), 'CYAN')}")
    if out_dir != base_out_dir:
        print(f"  base:      {colored(str(base_out_dir.resolve()), 'CYAN')}")
    print(f"  meta:      {colored(str(metadata_path.resolve()), 'CYAN')}")
    print(f"  resume:    {colored(str(processed_path.resolve()), 'CYAN')}")

    if aborted_reason is not None:
        raise SystemExit(f"Run aborted: {aborted_reason}")


if __name__ == "__main__":
    main()
