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


def config_to_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map run profile keys to argparse defaults."""
    defaults: Dict[str, Any] = {}

    run = cfg.get("run") or {}
    model = cfg.get("model") or {}
    vllm = cfg.get("vllm") or {}
    sampling = cfg.get("sampling") or cfg.get("generation") or {}
    prompt = cfg.get("prompt") or {}

    # run section
    _maybe_set(defaults, "dataset", run.get("dataset"))
    _maybe_set(defaults, "split", run.get("split"))
    _maybe_set(defaults, "out_dir", run.get("out_dir"))
    _maybe_set(defaults, "processed_ids", run.get("processed_ids"))
    _maybe_set(defaults, "batch_size", run.get("batch_size"))
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

    # vllm section
    _maybe_set(defaults, "tensor_parallel_size", vllm.get("tensor_parallel_size"))
    _maybe_set(defaults, "pipeline_parallel_size", vllm.get("pipeline_parallel_size"))
    _maybe_set(defaults, "data_parallel_size", vllm.get("data_parallel_size"))
    _maybe_set(defaults, "enable_expert_parallel", vllm.get("enable_expert_parallel"))
    _maybe_set(defaults, "max_model_len", vllm.get("max_model_len"))
    _maybe_set(defaults, "gpu_memory_utilization", vllm.get("gpu_memory_utilization"))
    _maybe_set(defaults, "enable_chunked_prefill", vllm.get("enable_chunked_prefill"))
    _maybe_set(defaults, "max_num_batched_tokens", vllm.get("max_num_batched_tokens"))
    _maybe_set(defaults, "max_num_seqs", vllm.get("max_num_seqs"))
    _maybe_set(defaults, "enable_prefix_caching", vllm.get("enable_prefix_caching"))
    _maybe_set(defaults, "kv_cache_dtype", vllm.get("kv_cache_dtype"))
    _maybe_set(defaults, "calculate_kv_scales", vllm.get("calculate_kv_scales"))
    _maybe_set(defaults, "quantization", vllm.get("quantization"))
    _maybe_set(defaults, "max_parallel_loading_workers", vllm.get("max_parallel_loading_workers"))
    _maybe_set(defaults, "dtype", vllm.get("dtype"))

    # sampling section
    _maybe_set(defaults, "temperature", sampling.get("temperature"))
    _maybe_set(defaults, "top_p", sampling.get("top_p"))
    _maybe_set(defaults, "max_new_tokens", sampling.get("max_new_tokens"))
    _maybe_set(defaults, "seed", sampling.get("seed"))
    _maybe_set(defaults, "structured_output", sampling.get("structured_output"))

    # prompt section
    _maybe_set(defaults, "disable_thinking", prompt.get("disable_thinking"))
    _maybe_set(defaults, "chat_template_kwargs", prompt.get("chat_template_kwargs"))

    return defaults
