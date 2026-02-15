# Change Checklist

## Schema Changes (`configs/schemas/schema.json`)
- Confirm field names are stable (downstream expects exact keys).
- Confirm array fields still have valid `items` typing.
- Run:
  - `uv run open-patients-test`
- Pay attention to:
  - `tests/test_schema_loader.py`
  - `tests/test_schema_loader_arrays.py`
  - `tests/test_prompts.py`
  - `tests/test_extraction.py`

## Prompt Changes (`src/core/prompts.py`, `src/utils/utils.py`)
- Ensure system prompt still includes enum/type constraints from schema.
- Verify chat-template fallback behavior still works (`prompt_mode=plain` path).
- Run:
  - `uv run open-patients-check-prompt --config configs/runs/medgemma-27b-text-it.yaml --seed 1`
  - `uv run open-patients-test`

## Worker/CLI Changes (`src/cli/*.py`, `src/core/config.py`)
- Preserve precedence: defaults < config profile < explicit CLI flags.
- Validate run directory and resume semantics.
- Validate replica launch args still forward correctly.
- Run:
  - `uv run open-patients-test`
  - Optional smoke:
    - `uv run open-patients-worker --config configs/runs/medgemma-27b-text-it.yaml --max_notes 5`
    - `uv run open-patients-replicas --config configs/runs/medgemma-27b-text-it.yaml --dry_run`

## Writer/Output Changes (`src/core/writer.py`, metadata in `src/cli/enrich.py`)
- Ensure shard naming and rollover behavior is unchanged unless intentionally modified.
- Ensure `processed_ids*.txt` still flushes/loads correctly.
- Ensure metadata JSON remains valid and includes expected counters.
- Run:
  - `uv run open-patients-test`
  - Inspect generated files in `outputs/...` from a short smoke run.

## vLLM Integration Changes (`src/core/llm_vllm.py`)
- Keep compatibility shims for differing vLLM param names (`max_tokens` vs `max_new_tokens`, structured output params).
- Verify fallback behavior when structured output is unsupported.
- Prefer dry-run or small benchmark validation before long jobs:
  - `uv run open-patients-bench --config configs/runs/medgemma-27b-text-it.yaml --max_notes 20`
