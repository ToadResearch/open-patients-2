#!/usr/bin/env python3
"""
Main entry point for the Open-Patients enrichment pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from tqdm import tqdm

from ..core.config import config_to_defaults, load_run_config
from ..core.extraction import derive_source_url, ensure_schema, load_usmle_id_to_row
from ..core.llm_vllm import build_llm, build_sampling
from ..core.prompts import USER_TEMPLATE, build_system_prompt
from ..core.schema_loader import load_schema
from ..core.utils import make_chat_prompt, now_iso, safe_json_extract, truncate_note
from ..core.writer import JSONLShardedWriter, load_processed_ids, ProcessedIdWriter


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default=None, help="Run profile YAML (configs/runs/*.yaml)"
    )
    cfg_args, remaining = config_parser.parse_known_args(argv)

    cfg = load_run_config(cfg_args.config) if cfg_args.config else {}
    defaults = config_to_defaults(cfg)

    ap = argparse.ArgumentParser(
        description="Enrich Open-Patients dataset with structured clinical fields."
    )
    ap.add_argument(
        "--config", default=cfg_args.config, help="Run profile YAML (configs/runs/*.yaml)"
    )
    ap.add_argument("--dataset", default="ncbi/Open-Patients", help="HF dataset name")
    ap.add_argument("--split", default="train", help="dataset split (Open-Patients uses train)")
    ap.add_argument("--model", default=None, help="vLLM model name or path")
    ap.add_argument(
        "--prompt_mode",
        default="chat",
        choices=["chat", "plain"],
        help="Prompt formatting mode (chat uses tokenizer template if available)",
    )
    ap.add_argument(
        "--prompt_style",
        default="verbose",
        choices=["compact", "verbose"],
        help="Prompt verbosity (compact avoids large enum lists)",
    )

    ap.add_argument("--out_dir", default=None, help="Output directory for enriched JSONL shards")
    ap.add_argument(
        "--processed_ids",
        default="processed_ids.txt",
        help="Resume marker file (in out_dir by default)",
    )
    ap.add_argument(
        "--schema",
        default="configs/schemas/schema.json",
        help="Path to JSON schema wrapper (default: configs/schemas/schema.json)",
    )
    ap.add_argument(
        "--usmle_mapping",
        default="configs/usmle_mapping.json",
        help="Path to configs/usmle_mapping.json (for usmle-<num> -> HF viewer row mapping)",
    )
    ap.add_argument(
        "--chat_template_kwargs",
        default=None,
        help="JSON dict of chat template kwargs (merged with --disable_thinking)",
    )
    ap.add_argument(
        "--run_id",
        default=None,
        help="Optional run id subfolder name (under out_dir)",
    )
    ap.add_argument(
        "--run_tag",
        default=None,
        help="Optional tag to prefix output shards/metadata (for multi-process runs)",
    )

    ap.add_argument("--batch_size", type=int, default=32, help="#prompts per llm.generate() call")
    ap.add_argument("--max_notes", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--max_input_chars", type=int, default=8000, help="truncate long notes for speed"
    )
    ap.add_argument("--max_new_tokens", type=int, default=700)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)

    # Parallelism / memory
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--pipeline_parallel_size", type=int, default=1)
    ap.add_argument("--data_parallel_size", type=int, default=1)

    ap.add_argument(
        "--enable_expert_parallel", action="store_true", help="Recommended for MoE models"
    )
    ap.add_argument("--max_model_len", type=int, default=8192, help="vLLM max context")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    ap.add_argument("--dtype", default="auto", help='vLLM dtype, e.g. "auto", "bf16", "fp16"')

    # Throughput knobs
    ap.add_argument("--enable_chunked_prefill", action="store_true", help="Enable chunked prefill")
    ap.add_argument("--max_num_batched_tokens", type=int, default=8192)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--enable_prefix_caching", action="store_true")

    # KV cache
    ap.add_argument("--kv_cache_dtype", default="fp8", help='e.g. "auto", "fp8"')
    ap.add_argument(
        "--calculate_kv_scales", action="store_true", help="Calibrate KV FP8 scales at warmup"
    )

    # Quantization override (optional; pre-quantized checkpoints often work with auto)
    ap.add_argument("--quantization", default=None, help='e.g. "marlin", "gptq", "awq", "fp8"')

    # Loading stability
    ap.add_argument(
        "--max_parallel_loading_workers", type=int, default=2, help="Avoid CPU RAM spikes on load"
    )

    ap.add_argument("--shard_size", type=int, default=50_000, help="JSONL records per output shard")
    ap.add_argument("--resume", action="store_true", help="Skip IDs already in processed_ids file")

    ap.add_argument("--structured_output", action="store_true", help="Use vLLM structured output")
    ap.add_argument(
        "--disable_thinking", action="store_true", help="Disable Qwen3-style <think> output"
    )

    # Manual dataset sharding across processes (CPU-side); useful for 2× replica mode
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)

    ap.set_defaults(**defaults)
    return ap.parse_args(remaining)


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("Missing --model (or model.name in --config).")
    if not args.out_dir:
        raise SystemExit("Missing --out_dir (or run.out_dir in --config).")

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

    processed_path = Path(args.processed_ids)
    if not processed_path.is_absolute():
        if args.run_tag:
            processed_path = out_dir / (
                f"{processed_path.stem}_{args.run_tag}{processed_path.suffix}"
            )
        else:
            processed_path = out_dir / processed_path

    processed_ids = load_processed_ids(processed_path) if args.resume else set()
    processed_writer = ProcessedIdWriter(processed_path, flush_every=2000)

    # Load USMLE row mapping (optional but recommended)
    usmle_map_path = Path(args.usmle_mapping)
    if not usmle_map_path.is_absolute():
        # treat as relative to current working directory
        usmle_map_path = Path.cwd() / usmle_map_path

    usmle_id_to_row = None
    if usmle_map_path.exists():
        usmle_id_to_row = load_usmle_id_to_row(usmle_map_path)
    else:
        usmle_id_to_row = None

    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = Path.cwd() / schema_path
    if not schema_path.exists():
        raise SystemExit(f"Schema not found: {schema_path}")

    schema_bundle = load_schema(schema_path)

    # vLLM init (filter kwargs for compatibility across vLLM versions)
    vllm_cfg = dict(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=args.dtype,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        kv_cache_dtype=args.kv_cache_dtype,
        calculate_kv_scales=args.calculate_kv_scales,
        quantization=args.quantization,
        max_parallel_loading_workers=args.max_parallel_loading_workers,
    )
    llm, tokenizer = build_llm(args.model, vllm_cfg)

    json_schema = schema_bundle.wrapper if args.structured_output else None
    sampling_cfg = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed if args.seed != 0 else None,
    )
    sampling = build_sampling(sampling_cfg, args.structured_output, json_schema)

    shard_prefix = f"data_shard_{args.run_tag}" if args.run_tag else "data_shard"
    writer = JSONLShardedWriter(
        out_dir=out_dir,
        shard_size=args.shard_size,
        name_prefix=shard_prefix,
    )
    keys_str = json.dumps(schema_bundle.schema_keys, ensure_ascii=False)

    chat_template_kwargs = {}
    if args.chat_template_kwargs:
        if isinstance(args.chat_template_kwargs, str):
            try:
                chat_template_kwargs.update(json.loads(args.chat_template_kwargs))
            except Exception as exc:  # pragma: no cover
                raise SystemExit(f"Invalid --chat_template_kwargs JSON: {exc}")
        elif isinstance(args.chat_template_kwargs, dict):
            chat_template_kwargs.update(args.chat_template_kwargs)
    if args.disable_thinking:
        # Qwen3-style tokenizers commonly support enable_thinking=False
        chat_template_kwargs["enable_thinking"] = False
    force_plain = args.prompt_mode == "plain"
    system_prompt = build_system_prompt(schema_bundle, style=args.prompt_style)

    buf_ids: List[str] = []
    buf_notes: List[str] = []
    buf_raw_rows: List[dict] = []

    n_total = 0
    n_written = 0
    n_skipped = 0
    n_failed = 0
    input_tokens = 0
    output_tokens = 0
    gen_time = 0.0

    start_iso = now_iso()
    start_perf = time.perf_counter()

    pbar = tqdm(total=args.max_notes if args.max_notes else None, desc="enrich")

    def _count_prompt_tokens(prompt: str) -> int:
        if hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(prompt))
            except Exception:
                return 0
        return 0

    def _count_output_tokens(out) -> int:
        if out.outputs and getattr(out.outputs[0], "token_ids", None) is not None:
            return len(out.outputs[0].token_ids)
        text = out.outputs[0].text if out.outputs else ""
        if hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(text))
            except Exception:
                return 0
        return 0

    def flush_batch() -> None:
        nonlocal n_written, n_failed, input_tokens, output_tokens, gen_time
        if not buf_notes:
            return

        prompts = []
        for note in buf_notes:
            note2 = truncate_note(note, args.max_input_chars)
            user = USER_TEMPLATE.format(note=note2, keys=keys_str)
            prompt = make_chat_prompt(
                tokenizer,
                system_prompt,
                user,
                chat_template_kwargs,
                force_plain=force_plain,
            )
            prompts.append(prompt)
            input_tokens += _count_prompt_tokens(prompt)

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        t1 = time.perf_counter()
        gen_time += (t1 - t0)

        created_at = now_iso()
        for _id, row, out in zip(buf_ids, buf_raw_rows, outputs):
            source_url = derive_source_url(_id, usmle_id_to_row)
            text = out.outputs[0].text if out.outputs else ""
            output_tokens += _count_output_tokens(out)

            parsed: Optional[dict]
            if args.structured_output:
                # Usually already JSON; try fast path first.
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = safe_json_extract(text)
            else:
                parsed = safe_json_extract(text)

            if parsed is None or not isinstance(parsed, dict):
                n_failed += 1
                rec = {
                    "id": _id,
                    "patient_note": row.get("description", ""),
                    "source": source_url,
                    "extraction_ok": False,
                    "model_output_raw": text,
                    "created_at": created_at,
                }
                writer.write(rec)
                processed_writer.add(_id)
                n_written += 1
                continue

            parsed = ensure_schema(
                parsed, schema_bundle.schema_keys, schema_bundle.typed_list_names
            )

            rec = {
                "id": _id,
                "patient_note": row.get("description", ""),
                "source": source_url,
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

        buf_ids.clear()
        buf_notes.clear()
        buf_raw_rows.clear()

    # Iterate stream
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    for row in ds:
        _id = row.get("_id")
        note = row.get("description", "")
        if not _id or not isinstance(note, str):
            continue

        n_total += 1

        # stable shard for multi-process dataset partitioning
        if args.num_shards > 1:
            h = int(hashlib.md5(_id.encode("utf-8")).hexdigest(), 16)
            if (h % args.num_shards) != args.shard_idx:
                continue

        if args.resume and _id in processed_ids:
            n_skipped += 1
            continue

        buf_ids.append(_id)
        buf_notes.append(note)
        buf_raw_rows.append(row)

        if len(buf_notes) >= args.batch_size:
            flush_batch()

        pbar.update(1)
        if args.max_notes and pbar.n >= args.max_notes:
            break

    flush_batch()
    writer.close()
    processed_writer.close()
    pbar.close()

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
        "resume": bool(args.resume),
        "config_path": args.config,
        "config": run_config,
        "schema_path": str(schema_path),
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_s": total_time,
        "gen_time_s": gen_time,
        "notes_seen": n_total,
        "notes_written": n_written,
        "notes_skipped": n_skipped,
        "notes_failed": n_failed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "notes_per_s": _safe_div(n_total, total_time),
        "input_toks_per_s": _safe_div(input_tokens, gen_time),
        "output_toks_per_s": _safe_div(output_tokens, gen_time),
        "total_toks_per_s": _safe_div(input_tokens + output_tokens, gen_time),
        "avg_input_toks_per_note": _safe_div(input_tokens, n_total),
        "avg_output_toks_per_note": _safe_div(output_tokens, n_total),
        "args": vars(args),
    }
    metadata_name = f"run_metadata_{args.run_tag}.json" if args.run_tag else "run_metadata.json"
    metadata_path = out_dir / metadata_name
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  seen:    {n_total}")
    print(f"  wrote:   {n_written}")
    print(f"  skipped: {n_skipped}")
    print(f"  failed:  {n_failed}")
    print(f"  out_dir: {out_dir.resolve()}")
    if out_dir != base_out_dir:
        print(f"  base:    {base_out_dir.resolve()}")
    print(f"  meta:    {metadata_path.resolve()}")
    print(f"  resume:  {processed_path.resolve()}")


if __name__ == "__main__":
    main()
