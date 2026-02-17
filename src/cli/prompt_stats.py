#!/usr/bin/env python3
"""
Compute prompt input-length distribution over Open-Patients notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from tqdm import tqdm

from ..core.config import config_to_defaults, load_run_config
from ..core.prompts import USER_TEMPLATE, build_system_prompt
from ..core.schema_loader import load_schema
from ..utils.utils import colored, make_chat_prompt, now_iso, print_header


def _parse_chat_template_kwargs(raw: object) -> dict:
    out = {}
    if not raw:
        return out
    if isinstance(raw, str):
        try:
            out.update(json.loads(raw))
        except Exception as exc:
            raise SystemExit(f"Invalid --chat_template_kwargs JSON: {exc}") from exc
    elif isinstance(raw, dict):
        out.update(raw)
    return out


def _load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "transformers is required to load tokenizers. "
            "Install with `uv add transformers` or `uv sync --extra vllm`."
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception as exc:
        raise SystemExit(f"Failed to load tokenizer '{name}': {exc}") from exc


def _percentile_nearest_rank(sorted_vals: List[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    n = len(sorted_vals)
    rank = int(math.ceil((pct / 100.0) * n))
    rank = min(max(rank, 1), n)
    return sorted_vals[rank - 1]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    base_defaults = {
        "dataset": "ncbi/Open-Patients",
        "split": "train",
        "schema": "configs/schemas/schema.json",
        "prompt_mode": "chat",
        "schema_in_prompt": False,
        "chat_template_kwargs": None,
        "disable_thinking": False,
        "max_notes": 0,
        "batch_size": 64,
        "num_shards": 1,
        "shard_idx": 0,
        "emit_lengths": False,
        "no_progress": False,
        "progress_path": None,
        "quiet": False,
        "json_out": None,
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

    ap = argparse.ArgumentParser(
        description="Measure rendered prompt length distribution across Open-Patients notes."
    )
    ap.add_argument("--config", help="Run profile YAML (configs/runs/*.yaml)")
    ap.add_argument("--dataset", help="HF dataset name")
    ap.add_argument("--split", help="Dataset split (Open-Patients uses train)")
    ap.add_argument("--schema", help="Path to JSON schema wrapper")
    ap.add_argument(
        "--prompt_mode",
        choices=["chat", "plain"],
        help="Prompt formatting mode (chat uses tokenizer template if available)",
    )
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="Hugging Face tokenizer name (e.g., Qwen/Qwen3.5-397B-A17B)",
    )
    ap.add_argument(
        "--schema_in_prompt",
        action="store_true",
        help="Embed full JSON schema wrapper in system prompt",
    )
    ap.add_argument(
        "--chat_template_kwargs",
        help="JSON dict of tokenizer chat template kwargs (merged with --disable_thinking)",
    )
    ap.add_argument("--disable_thinking", action="store_true")
    ap.add_argument("--max_notes", type=int, help="0 = all notes")
    ap.add_argument("--batch_size", type=int, help="Prompts per tokenizer batch")
    ap.add_argument("--num_shards", type=int, help="Total deterministic dataset shards")
    ap.add_argument("--shard_idx", type=int, help="Shard index to process (0-based)")
    ap.add_argument(
        "--emit_lengths",
        action="store_true",
        help="Include per-prompt token/char arrays in JSON output (used for aggregation)",
    )
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar")
    ap.add_argument(
        "--progress_path",
        help="Optional JSON file path for progress updates (used by replicas launcher)",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress end-of-run summary print")
    ap.add_argument(
        "--json_out",
        help="Optional path to write summary JSON (defaults to benchmarks/prompts/prompt_stats_*.json)",
    )

    ap.set_defaults(**defaults)
    return ap.parse_args(remaining)


def main() -> None:
    args = parse_args()
    print_header("Open-Patients Prompt Stats")
    if args.config:
        print(f"Config: {colored(args.config, 'CYAN')}")
    print(f"Tokenizer: {colored(args.tokenizer, 'CYAN')}")
    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise SystemExit("--shard_idx must satisfy 0 <= shard_idx < num_shards")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")

    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = Path.cwd() / schema_path
    if not schema_path.exists():
        raise SystemExit(f"Schema not found: {schema_path}")

    if args.json_out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.json_out = str(Path("benchmarks/prompts") / f"prompt_stats_{ts}.json")

    tokenizer = _load_tokenizer(args.tokenizer)
    schema_bundle = load_schema(schema_path)
    system_prompt = build_system_prompt(
        schema_bundle,
        include_json_schema=bool(args.schema_in_prompt),
    )
    keys_str = json.dumps(schema_bundle.schema_keys, ensure_ascii=False)

    chat_template_kwargs = _parse_chat_template_kwargs(args.chat_template_kwargs)
    if args.disable_thinking:
        chat_template_kwargs["enable_thinking"] = False
    force_plain = args.prompt_mode == "plain"

    lengths_tokens: List[int] = []
    lengths_chars: List[int] = []
    n_rows = 0
    n_skipped = 0

    start_iso = now_iso()
    t0 = time.perf_counter()
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    pbar = None if args.no_progress else tqdm(
        total=args.max_notes if args.max_notes else None,
        desc="prompt-stats",
    )
    prompt_buf: List[str] = []
    progress_path = Path(args.progress_path) if args.progress_path else None
    last_progress_write = 0.0

    def write_progress(done: bool = False, force: bool = False) -> None:
        nonlocal last_progress_write
        if progress_path is None:
            return
        now = time.time()
        if (not force) and (now - last_progress_write < 0.5):
            return
        payload = {
            "notes_processed": n_rows,
            "notes_skipped": n_skipped,
            "done": bool(done),
            "updated_at": now,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = progress_path.with_name(progress_path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(progress_path)
        last_progress_write = now

    write_progress(force=True)

    def flush_batch() -> None:
        if not prompt_buf:
            return
        encoded = tokenizer(
            prompt_buf,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_ids_batch = encoded.get("input_ids", [])
        if len(token_ids_batch) != len(prompt_buf):
            raise SystemExit(
                "Tokenizer returned mismatched batch lengths while computing prompt stats."
            )
        for prompt, token_ids in zip(prompt_buf, token_ids_batch):
            lengths_chars.append(len(prompt))
            lengths_tokens.append(len(token_ids))
        prompt_buf.clear()
        write_progress()

    for row in ds:
        row_id = row.get("_id") or row.get("id")
        if args.num_shards > 1:
            if not row_id:
                n_skipped += 1
                continue
            h = int(hashlib.md5(str(row_id).encode("utf-8")).hexdigest(), 16)
            if (h % args.num_shards) != args.shard_idx:
                continue

        note = row.get("description", "")
        if not isinstance(note, str):
            n_skipped += 1
            continue

        user_prompt = USER_TEMPLATE.format(note=note, keys=keys_str)
        prompt = make_chat_prompt(
            tokenizer,
            system_prompt,
            user_prompt,
            chat_template_kwargs=chat_template_kwargs,
            force_plain=force_plain,
        )
        prompt_buf.append(prompt)
        n_rows += 1

        if len(prompt_buf) >= args.batch_size:
            flush_batch()

        if pbar is not None:
            pbar.update(1)
        write_progress()
        if args.max_notes and n_rows >= args.max_notes:
            break

    flush_batch()

    if pbar is not None:
        pbar.close()
    duration_s = time.perf_counter() - t0
    end_iso = now_iso()

    if not lengths_tokens:
        raise SystemExit("No valid prompt lengths collected.")

    toks_sorted = sorted(lengths_tokens)
    chars_sorted = sorted(lengths_chars)

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "schema_path": str(schema_path),
        "tokenizer": args.tokenizer,
        "prompt_mode": args.prompt_mode,
        "schema_in_prompt": bool(args.schema_in_prompt),
        "max_notes": int(args.max_notes),
        "batch_size": int(args.batch_size),
        "num_shards": int(args.num_shards),
        "shard_idx": int(args.shard_idx),
        "notes_processed": n_rows,
        "notes_skipped": n_skipped,
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_s": duration_s,
        "tokens": {
            "min": toks_sorted[0],
            "p50": _percentile_nearest_rank(toks_sorted, 50.0),
            "p95": _percentile_nearest_rank(toks_sorted, 95.0),
            "p99": _percentile_nearest_rank(toks_sorted, 99.0),
            "max": toks_sorted[-1],
            "mean": sum(toks_sorted) / len(toks_sorted),
        },
        "chars": {
            "min": chars_sorted[0],
            "p50": _percentile_nearest_rank(chars_sorted, 50.0),
            "p95": _percentile_nearest_rank(chars_sorted, 95.0),
            "p99": _percentile_nearest_rank(chars_sorted, 99.0),
            "max": chars_sorted[-1],
            "mean": sum(chars_sorted) / len(chars_sorted),
        },
    }
    if args.emit_lengths:
        metrics["token_lengths"] = lengths_tokens
        metrics["char_lengths"] = lengths_chars

    out_path = Path(args.json_out)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = Path("benchmarks/prompts") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_progress(done=True, force=True)

    if not args.quiet:
        print("")
        print(colored("Token length summary", "CYAN"))
        print(f"  notes: {colored(str(n_rows), 'GREEN')}")
        print(
            f"  notes/s:{colored(f'{(n_rows / duration_s) if duration_s > 0 else 0.0:.2f}', 'GREEN')}"
        )
        print(f"  p50:   {colored(str(metrics['tokens']['p50']), 'GREEN')}")
        print(f"  p95:   {colored(str(metrics['tokens']['p95']), 'GREEN')}")
        print(f"  p99:   {colored(str(metrics['tokens']['p99']), 'GREEN')}")
        print(f"  max:   {colored(str(metrics['tokens']['max']), 'GREEN')}")
        token_mean = f"{metrics['tokens']['mean']:.2f}"
        print(f"  mean:  {colored(token_mean, 'GREEN')}")
        print(f"\n{colored('Wrote summary:', 'CYAN')} {colored(str(out_path), 'CYAN')}")


if __name__ == "__main__":
    main()
