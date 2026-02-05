#!/usr/bin/env python3
"""Benchmark throughput for a given run profile."""

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
from ..core.llm_vllm import build_llm, build_sampling
from ..core.prompts import USER_TEMPLATE, build_system_prompt
from ..core.schema_loader import load_schema
from ..core.utils import make_chat_prompt, now_iso, truncate_note


def _apply_benchmark_defaults(defaults: dict, cfg: dict) -> dict:
    bench = cfg.get("benchmark") or {}
    out = dict(defaults)

    # Always override max_notes for bench unless explicitly provided in benchmark.
    out["max_notes"] = bench.get("max_notes", 500)

    if "batch_size" in bench:
        out["batch_size"] = bench.get("batch_size")
    if "max_input_chars" in bench:
        out["max_input_chars"] = bench.get("max_input_chars")
    if "max_new_tokens" in bench:
        out["max_new_tokens"] = bench.get("max_new_tokens")

    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default=None, help="Run profile YAML (configs/runs/*.yaml)"
    )
    cfg_args, remaining = config_parser.parse_known_args(argv)

    cfg = load_run_config(cfg_args.config) if cfg_args.config else {}
    defaults = config_to_defaults(cfg)
    defaults = _apply_benchmark_defaults(defaults, cfg)

    ap = argparse.ArgumentParser(description="Benchmark Open-Patients enrichment throughput.")

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

    ap.add_argument(
        "--schema", default="configs/schemas/schema.json", help="Path to JSON schema wrapper"
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--max_notes", type=int, default=500, help="Number of notes to benchmark (0 = all)"
    )
    ap.add_argument("--max_input_chars", type=int, default=8000)
    ap.add_argument("--max_new_tokens", type=int, default=700)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)

    # vLLM args
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--pipeline_parallel_size", type=int, default=1)
    ap.add_argument("--data_parallel_size", type=int, default=1)
    ap.add_argument("--enable_expert_parallel", action="store_true")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--enable_chunked_prefill", action="store_true")
    ap.add_argument("--max_num_batched_tokens", type=int, default=8192)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--enable_prefix_caching", action="store_true")
    ap.add_argument("--kv_cache_dtype", default="fp8")
    ap.add_argument("--calculate_kv_scales", action="store_true")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--max_parallel_loading_workers", type=int, default=2)

    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--run_tag", default=None, help="Optional tag for multi-process benchmarks")

    ap.add_argument("--structured_output", action="store_true")
    ap.add_argument("--disable_thinking", action="store_true")
    ap.add_argument(
        "--chat_template_kwargs", default=None, help="JSON dict of chat template kwargs"
    )

    ap.add_argument(
        "--json_out",
        default=None,
        help="Write metrics to JSON file (defaults to benchmarks/bench_*.json)",
    )

    ap.set_defaults(**defaults)
    return ap.parse_args(remaining)


def _count_prompt_tokens(tokenizer, prompt: str) -> int:
    if hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(prompt))
        except Exception:
            return 0
    return 0


def _count_output_tokens(tokenizer, out) -> int:
    if out.outputs and getattr(out.outputs[0], "token_ids", None) is not None:
        return len(out.outputs[0].token_ids)
    text = out.outputs[0].text if out.outputs else ""
    if hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text))
        except Exception:
            return 0
    return 0


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("Missing --model (or model.name in --config).")

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

    chat_template_kwargs = {}
    if args.chat_template_kwargs:
        if isinstance(args.chat_template_kwargs, str):
            try:
                chat_template_kwargs.update(json.loads(args.chat_template_kwargs))
            except Exception as exc:
                raise SystemExit(f"Invalid --chat_template_kwargs JSON: {exc}")
        elif isinstance(args.chat_template_kwargs, dict):
            chat_template_kwargs.update(args.chat_template_kwargs)
    if args.disable_thinking:
        chat_template_kwargs["enable_thinking"] = False
    force_plain = args.prompt_mode == "plain"
    system_prompt = build_system_prompt(schema_bundle, style=args.prompt_style)
    keys_str = json.dumps(schema_bundle.schema_keys, ensure_ascii=False)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    buf_notes: List[str] = []
    n_total = 0
    n_batches = 0
    input_tokens = 0
    output_tokens = 0
    gen_time = 0.0

    start_iso = now_iso()
    total_start = time.perf_counter()
    pbar = tqdm(total=args.max_notes if args.max_notes else None, desc="bench")

    def flush_batch() -> None:
        nonlocal n_batches, input_tokens, output_tokens, gen_time
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
            input_tokens += _count_prompt_tokens(tokenizer, prompt)

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        t1 = time.perf_counter()
        gen_time += t1 - t0
        n_batches += 1

        for out in outputs:
            output_tokens += _count_output_tokens(tokenizer, out)

        buf_notes.clear()

    for row in ds:
        row_id = row.get("_id") or row.get("id")
        note = row.get("description", "")
        if not isinstance(note, str):
            continue

        if args.num_shards > 1:
            if not row_id:
                continue
            h = int(hashlib.md5(str(row_id).encode("utf-8")).hexdigest(), 16)
            if (h % args.num_shards) != args.shard_idx:
                continue

        n_total += 1
        buf_notes.append(note)

        if len(buf_notes) >= args.batch_size:
            flush_batch()

        pbar.update(1)
        if args.max_notes and n_total >= args.max_notes:
            break

    flush_batch()
    pbar.close()
    total_time = time.perf_counter() - total_start
    end_iso = now_iso()

    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    metrics = {
        "model": args.model,
        "config": args.config,
        "run_tag": args.run_tag,
        "max_notes": args.max_notes,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_style": args.prompt_style,
        "prompt_mode": args.prompt_mode,
        "structured_output": bool(args.structured_output),
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "start_time": start_iso,
        "end_time": end_iso,
        "notes": n_total,
        "batches": n_batches,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "total_time_s": total_time,
        "gen_time_s": gen_time,
        "notes_per_s": _safe_div(n_total, total_time),
        "input_toks_per_s": _safe_div(input_tokens, gen_time),
        "output_toks_per_s": _safe_div(output_tokens, gen_time),
        "total_toks_per_s": _safe_div(input_tokens + output_tokens, gen_time),
        "avg_input_toks_per_note": _safe_div(input_tokens, n_total),
        "avg_output_toks_per_note": _safe_div(output_tokens, n_total),
    }

    print("\nBenchmark results")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute() and out_path.parent == Path("."):
            out_path = Path("benchmarks") / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to: {out_path}")


if __name__ == "__main__":
    main()
