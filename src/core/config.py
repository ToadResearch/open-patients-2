"""
Run profile (YAML) loading and mapping to CLI defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_run_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML run profile. Returns an empty dict if path is None."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top level: {p}")
    return data


def _maybe_set(defaults: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        defaults[key] = value


def _samples_to_max_notes(value: Any) -> int:
    """
    Map run.samples to existing max_notes semantics.

    - omitted / null / negative => 0 (process all notes)
    - positive integer => that many notes
    """
    if value is None:
        return 0
    try:
        n = int(value)
    except Exception as exc:
        raise ValueError(f"run.samples must be an integer, got: {value!r}") from exc
    if n < 0:
        return 0
    return n


def config_to_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map run profile keys to argparse defaults."""
    defaults: Dict[str, Any] = {}

    run = cfg.get("run") or {}
    model = cfg.get("model") or {}
    api = cfg.get("api") or {}
    sampling = cfg.get("sampling") or cfg.get("generation") or {}
    prompt = cfg.get("prompt") or {}

    # run section
    _maybe_set(defaults, "dataset", run.get("dataset"))
    _maybe_set(defaults, "split", run.get("split"))
    _maybe_set(defaults, "out_dir", run.get("out_dir"))
    _maybe_set(defaults, "processed_ids", run.get("processed_ids"))
    _maybe_set(defaults, "batch_size", run.get("batch_size"))
    if "samples" in run:
        _maybe_set(defaults, "max_notes", _samples_to_max_notes(run.get("samples")))
    else:
        _maybe_set(defaults, "max_notes", run.get("max_notes"))
    _maybe_set(defaults, "shard_size", run.get("shard_size"))
    _maybe_set(defaults, "resume", run.get("resume"))
    _maybe_set(defaults, "num_shards", run.get("num_shards"))
    _maybe_set(defaults, "shard_idx", run.get("shard_idx"))
    _maybe_set(defaults, "usmle_mapping", run.get("usmle_mapping"))
    _maybe_set(defaults, "schema", run.get("schema"))
    if "structured_output" in run and "structured_output" not in sampling:
        _maybe_set(defaults, "structured_output", run.get("structured_output"))

    # model section
    _maybe_set(defaults, "model", model.get("name"))
    _maybe_set(defaults, "prompt_mode", model.get("prompt_mode"))

    # api section
    _maybe_set(defaults, "api_timeout_s", api.get("timeout_s"))
    _maybe_set(defaults, "api_max_retries", api.get("max_retries"))
    _maybe_set(defaults, "api_retry_backoff_initial_s", api.get("retry_backoff_initial_s"))
    _maybe_set(defaults, "api_retry_backoff_max_s", api.get("retry_backoff_max_s"))
    _maybe_set(defaults, "api_outage_abort_after_s", api.get("outage_abort_after_s"))
    _maybe_set(defaults, "api_endpoints", api.get("endpoints"))

    # sampling section
    _maybe_set(defaults, "temperature", sampling.get("temperature"))
    _maybe_set(defaults, "top_p", sampling.get("top_p"))
    _maybe_set(defaults, "max_new_tokens", sampling.get("max_new_tokens"))
    _maybe_set(defaults, "seed", sampling.get("seed"))
    _maybe_set(defaults, "structured_output", sampling.get("structured_output"))

    # prompt section
    _maybe_set(defaults, "disable_thinking", prompt.get("disable_thinking"))
    _maybe_set(defaults, "chat_template_kwargs", prompt.get("chat_template_kwargs"))
    _maybe_set(defaults, "schema_in_prompt", prompt.get("schema_in_prompt"))

    return defaults
