#!/usr/bin/env python3
"""
Inspect the prompt as a model would see it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset

from ..core.config import config_to_defaults, load_run_config
from ..core.prompts import USER_TEMPLATE, build_system_prompt
from ..core.schema_loader import load_schema
from ..utils.utils import colored, make_chat_prompt, print_header


def _parse_tokenizer_list(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    if not values:
        return out
    for v in values:
        for part in v.split(","):
            name = part.strip()
            if name:
                out.append(name)
    # de-dup while preserving order
    seen = set()
    deduped = []
    for name in out:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


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


def _select_row(ds, note_id: Optional[str], seed: Optional[int]) -> dict:
    if note_id:
        for row in ds:
            row_id = row.get("_id") or row.get("id")
            if row_id == note_id:
                return row
        raise SystemExit(f"Could not find _id={note_id} in dataset.")

    if seed is not None:
        random.seed(seed)
        ds = ds.shuffle(buffer_size=1000, seed=seed)
    else:
        ds = ds.shuffle(buffer_size=1000)

    for row in ds:
        note = row.get("description", "")
        if isinstance(note, str) and note.strip():
            return row
    raise SystemExit("No valid notes found in dataset.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    base_defaults = {
        "dataset": "ncbi/Open-Patients",
        "split": "train",
        "model": None,
        "schema": "configs/schemas/schema.json",
        "prompt_mode": "chat",
        "chat_template_kwargs": None,
        "disable_thinking": False,
        "id": None,
        "seed": None,
        "tokenizer": None,
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

    ap = argparse.ArgumentParser(description="Inspect the rendered prompt for a sample note.")
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
        "--id",
        dest="id",
        help="Use a specific Open-Patients _id (otherwise random)",
    )
    ap.add_argument("--seed", type=int, help="Random seed for sampling")
    ap.add_argument(
        "--model",
        help="Default tokenizer/model name if --tokenizer is not provided",
    )
    ap.add_argument(
        "--tokenizer",
        action="append",
        help="Tokenizer model name(s). Can be repeated or comma-separated.",
    )
    ap.add_argument(
        "--chat_template_kwargs",
        help="JSON dict of chat template kwargs (merged with --disable_thinking)",
    )
    ap.add_argument("--disable_thinking", action="store_true")

    ap.set_defaults(**defaults)
    return ap.parse_args(remaining)


def _render_prompt(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    prompt_mode: str,
    chat_template_kwargs: dict,
) -> str:
    force_plain = prompt_mode == "plain"
    return make_chat_prompt(
        tokenizer,
        system_prompt,
        user_prompt,
        chat_template_kwargs,
        force_plain=force_plain,
    )


def _print_counts(prompt: str, tokenizer, name: Optional[str]) -> None:
    chars = len(prompt)
    lines = len(prompt.splitlines()) or 1
    print(colored("Counts", "GREEN"))
    print(f"- chars: {colored(str(chars), 'GREEN')}")
    print(f"- lines: {colored(str(lines), 'GREEN')}")
    if tokenizer is not None:
        try:
            tokens = len(tokenizer.encode(prompt))
            label = name or "tokenizer"
            print(f"- tokens ({label}): {colored(str(tokens), 'GREEN')}")
        except Exception:
            label = name or "tokenizer"
            print(f"- tokens ({label}): {colored('<failed to encode>', 'RED')}")


def main() -> None:
    args = parse_args()

    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = Path.cwd() / schema_path
    if not schema_path.exists():
        raise SystemExit(f"Schema not found: {schema_path}")

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    row = _select_row(ds, args.id, args.seed)
    row_id = row.get("_id") or row.get("id")
    note = row.get("description", "")
    if not isinstance(note, str):
        raise SystemExit("Selected row does not contain a text description.")

    schema_bundle = load_schema(schema_path)
    system_prompt = build_system_prompt(schema_bundle)
    keys_str = json.dumps(schema_bundle.schema_keys, ensure_ascii=False)
    user_prompt = USER_TEMPLATE.format(note=note, keys=keys_str)

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

    tokenizer_names = _parse_tokenizer_list(args.tokenizer)
    if not tokenizer_names and args.model:
        tokenizer_names = [args.model]

    print_header("Prompt Check")
    print(f"Selected id: {colored(str(row_id), 'CYAN')}")
    print(f"Dataset: {colored(f'{args.dataset} ({args.split})', 'CYAN')}")
    print(f"Prompt mode: {colored(args.prompt_mode, 'CYAN')}")
    print(f"Note chars: {colored(str(len(note)), 'CYAN')}")
    print("")

    if not tokenizer_names:
        prompt = _render_prompt(
            None, system_prompt, user_prompt, args.prompt_mode, chat_template_kwargs
        )
        print(colored("Prompt", "MAGENTA"))
        print(colored(prompt, "WHITE"))
        print("")
        _print_counts(prompt, None, None)
        return

    for idx, name in enumerate(tokenizer_names, start=1):
        tokenizer = _load_tokenizer(name)
        prompt = _render_prompt(
            tokenizer, system_prompt, user_prompt, args.prompt_mode, chat_template_kwargs
        )
        print(
            colored(
                f"Prompt ({idx}/{len(tokenizer_names)}): {name}",
                "MAGENTA",
            )
        )
        print(colored(prompt, "WHITE"))
        print("")
        _print_counts(prompt, tokenizer, name)
        if idx != len(tokenizer_names):
            print("")


if __name__ == "__main__":
    main()
