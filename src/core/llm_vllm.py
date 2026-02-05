"""
vLLM construction helpers (engine + sampling) with compatibility shims.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams


def _require_vllm():
    try:
        import vllm  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("vLLM is not installed. Install with: uv sync --extra vllm") from exc
    return vllm


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Only pass kwargs that exist in the installed vLLM version."""
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if (v is not None and k in sig.parameters)}


def build_llm(model: str, vllm_cfg: Dict[str, Any]) -> Tuple["LLM", Any]:
    """Construct a vLLM engine and return (llm, tokenizer)."""
    vllm = _require_vllm()
    LLM = vllm.LLM
    llm_kwargs = dict(
        model=model,
        tensor_parallel_size=vllm_cfg.get("tensor_parallel_size"),
        pipeline_parallel_size=vllm_cfg.get("pipeline_parallel_size"),
        data_parallel_size=vllm_cfg.get("data_parallel_size"),
        enable_expert_parallel=vllm_cfg.get("enable_expert_parallel"),
        max_model_len=vllm_cfg.get("max_model_len"),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization"),
        trust_remote_code=vllm_cfg.get("trust_remote_code", True),
        dtype=vllm_cfg.get("dtype", "auto"),
        enable_chunked_prefill=vllm_cfg.get("enable_chunked_prefill"),
        max_num_batched_tokens=vllm_cfg.get("max_num_batched_tokens"),
        max_num_seqs=vllm_cfg.get("max_num_seqs"),
        enable_prefix_caching=vllm_cfg.get("enable_prefix_caching"),
        kv_cache_dtype=vllm_cfg.get("kv_cache_dtype"),
        calculate_kv_scales=vllm_cfg.get("calculate_kv_scales"),
        quantization=vllm_cfg.get("quantization"),
        max_parallel_loading_workers=vllm_cfg.get("max_parallel_loading_workers"),
    )
    llm = LLM(**_filter_kwargs(LLM.__init__, llm_kwargs))  # type: ignore[arg-type]
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def _build_structured_params(cls: Any, inner_schema: Dict[str, Any]) -> Any:
    """Instantiate structured output params across vLLM versions."""
    sig = inspect.signature(cls.__init__)
    if "json" in sig.parameters:
        return cls(json=inner_schema)
    if "json_schema" in sig.parameters:
        return cls(json_schema=inner_schema)
    if "schema" in sig.parameters:
        return cls(schema=inner_schema)
    return cls(inner_schema)


def build_sampling(
    sampling_cfg: Dict[str, Any],
    structured_output: bool,
    json_schema: Optional[Dict[str, Any]],
) -> "SamplingParams":
    """Construct SamplingParams with optional structured output constraints."""
    vllm = _require_vllm()
    SamplingParams = vllm.SamplingParams

    StructuredOutputsParams = None
    GuidedDecodingParams = None
    try:
        from vllm import sampling_params as vllm_sampling_params  # type: ignore

        StructuredOutputsParams = getattr(vllm_sampling_params, "StructuredOutputsParams", None)
        GuidedDecodingParams = getattr(vllm_sampling_params, "GuidedDecodingParams", None)
    except Exception:  # pragma: no cover
        StructuredOutputsParams = None
        GuidedDecodingParams = None

    if StructuredOutputsParams is None:
        StructuredOutputsParams = getattr(vllm, "StructuredOutputsParams", None)
    if GuidedDecodingParams is None:
        GuidedDecodingParams = getattr(vllm, "GuidedDecodingParams", None)

    sampling_kwargs = dict(
        temperature=sampling_cfg.get("temperature"),
        top_p=sampling_cfg.get("top_p"),
        max_tokens=sampling_cfg.get("max_new_tokens"),
        seed=sampling_cfg.get("seed"),
    )

    if structured_output and json_schema is not None:
        inner = json_schema["json_schema"]["schema"]
        sig = inspect.signature(SamplingParams.__init__)
        if "structured_outputs" in sig.parameters and StructuredOutputsParams is not None:
            sampling_kwargs["structured_outputs"] = _build_structured_params(
                StructuredOutputsParams, inner
            )
        elif "guided_decoding" in sig.parameters and GuidedDecodingParams is not None:
            sampling_kwargs["guided_decoding"] = _build_structured_params(
                GuidedDecodingParams, inner
            )
        elif "guided_decoding_params" in sig.parameters and GuidedDecodingParams is not None:
            sampling_kwargs["guided_decoding_params"] = _build_structured_params(
                GuidedDecodingParams, inner
            )
        elif "json_schema" in sig.parameters:
            sampling_kwargs["json_schema"] = inner
        else:
            print(
                "[warn] structured_output requested but unsupported by this vLLM version; "
                "falling back to unconstrained generation."
            )

    return SamplingParams(**_filter_kwargs(SamplingParams.__init__, sampling_kwargs))  # type: ignore[arg-type]
