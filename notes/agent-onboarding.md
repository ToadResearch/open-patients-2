# Agent Onboarding

## What This Repo Does
Transforms `ncbi/Open-Patients` notes into schema-constrained structured JSONL records with local vLLM inference.

## 10-Minute Orientation
1. Read `README.md` (quickstart + outputs).
2. Read your target run profile in `configs/runs/`.
3. Read `src/cli/enrich.py` end-to-end.
4. Read `src/core/schema_loader.py`, `src/core/prompts.py`, `src/core/extraction.py`.
5. Skim tests in `tests/` to understand invariants.

## Mental Model
- Input: streaming HF dataset rows (`_id`, `description`).
- Prompting: schema-driven system prompt + note-specific user template.
- Generation: vLLM, optionally constrained by JSON schema (`--structured_output`).
- Normalization: enforce schema keys, list defaults, and failure fallback record.
- Output: sharded JSONL + processed IDs + metadata JSON.

## First Commands To Run
- `uv run open-patients-test`
- `uv run open-patients-check-prompt --config configs/runs/medgemma-27b-text-it.yaml --seed 7`
- If doing runtime work on GPU:
  - `uv run open-patients-worker --config configs/runs/medgemma-27b-text-it.yaml --max_notes 10`

## Common Pitfalls
- Forgetting config precedence: CLI flags override profile values.
- Confusing `resume` behavior with run directory creation.
- Breaking list/scalar classification when editing schema fields.
- Editing prompt logic without re-checking tokenizer-template behavior.
- Replica runs writing to overlapping files when `run_tag` is missing.

## Where To Look For Issues
- Missing fields or malformed output: `src/core/extraction.py`, `src/core/schema_loader.py`
- Bad prompt format: `src/core/prompts.py`, `src/utils/utils.py::make_chat_prompt`
- Slow/unstable inference: `src/core/llm_vllm.py`, run profile `vllm:` settings
- Resume/output collisions: `src/core/writer.py`, `src/cli/enrich.py`, `src/cli/launch.py`
