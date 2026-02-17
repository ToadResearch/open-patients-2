# AGENTS.md

## Purpose
Open-Patients+ enriches `ncbi/Open-Patients` clinical notes into structured JSONL using a local vLLM model and a JSON schema.

Outputs are written as sharded JSONL under `out_dir/shards/` and (by default) a combined `out_dir/data.jsonl`.

## Read First
1. `README.md`
2. `configs/runs/*.yaml` (active run profile)
3. `src/cli/enrich.py` (main pipeline)
4. `src/core/schema_loader.py`, `src/core/prompts.py`, `src/core/extraction.py`

## Project Map
- `src/cli/enrich.py`: main worker; streams dataset, builds prompts, runs generation, writes shards under `out_dir/shards/`, and optionally combines into `out_dir/data.jsonl`.
- `src/cli/launch.py`: multi-GPU replica launcher for worker runs.
- `src/cli/bench.py`: throughput benchmark (single process).
- `src/cli/bench_replicas.py`: multi-GPU benchmark launcher + aggregate metrics.
- `src/cli/check_prompt.py`: renders prompt for sample note/tokenizer(s).
- `src/cli/usmle_map.py`: idempotent wrapper for generating `configs/usmle_mapping.json`.
- `src/cli/push_to_hf.py`: uploads output dataset to Hugging Face.
- `src/core/config.py`: YAML run-profile loading + mapping profile values to CLI defaults.
- `src/core/llm_vllm.py`: vLLM construction with compatibility shims across vLLM versions.
- `src/core/schema_loader.py`: loads schema wrapper and derives scalar/list field metadata.
- `src/core/prompts.py`: schema-driven system prompt + user template.
- `src/core/extraction.py`: output normalization and source URL derivation.
- `src/core/writer.py`: sharded JSONL writer and resume id tracking.
- `tests/`: `unittest` suite covering config mapping, schema parsing, prompt rules, writer behavior, and extraction helpers.

## Runtime Flow (Worker)
1. Parse args in `src/cli/enrich.py` with defaults + optional run profile (`--config`) + CLI overrides.
2. Resolve `out_dir`:
   - absolute path: unchanged
   - relative path not starting with `outputs/`: auto-prefixed with `outputs/`
   - recommended convention: `./open_patients_<model_slug>` (e.g., `./open_patients_medgemma1_5_4b_unsloth`)
3. Create run directory when `--resume` is false (auto `run_YYYYmmdd_HHMMSS_xxxxxx` unless `--run_id` provided).
4. Load schema bundle from `configs/schemas/schema.json` (or `--schema`).
5. Build system prompt from schema and render per-note prompt via tokenizer chat template (or plain mode).
6. Generate via vLLM; optionally enforce structured output.
7. Parse/normalize outputs; write JSONL shards under `out_dir/shards/`, plus `processed_ids*.txt` and `run_metadata*.json`.
8. Finalize outputs:
   - single-process worker: concatenates shards into `out_dir/data.jsonl` (disable via `--no_combine_shards`)
   - replica runs: `open-patients-replicas` concatenates all shard files into `out_dir/data.jsonl` after workers finish

## Important Contracts
- Schema is the source of truth for output keys (`load_schema`).
- `ensure_schema` drops unknown keys, fills missing scalars with `null`, and enforces list fields as lists.
- On parse failure, records are still written with:
  - `extraction_ok: false`
  - `model_output_raw`
- Shards are written to `out_dir/shards/*.jsonl`; `data.jsonl` is a simple concatenation (no global sorting guarantee).
- Resume tracking is file-based (`processed_ids*.txt`).
- Replica runs rely on deterministic hash sharding (`--num_shards`, `--shard_idx`) and distinct `--run_tag`.

## Common Commands
- Setup:
  - `uv venv && source .venv/bin/activate && uv sync`
  - Linux + GPU: `uv sync --extra vllm`
- Generate mapping if needed:
  - `uv run open-patients-usmle-map`
- Run worker:
  - `uv run open-patients-worker --config configs/runs/medgemma-27b-text-it-unsloth.yaml`
- Run multi-GPU replicas:
  - `uv run open-patients-replicas --config configs/runs/medgemma-27b-text-it-unsloth.yaml --gpus 0,1,2,3`
- Prompt inspection:
  - `uv run open-patients-check-prompt --config configs/runs/medgemma-27b-text-it-unsloth.yaml --seed 1`
- Bench:
  - `uv run open-patients-bench --config configs/runs/medgemma-27b-text-it-unsloth.yaml`
- Sweep (single GPU):
  - `.venv/bin/python scripts/sweep_single_h100.py --config configs/runs/medgemma-1.5-4b-it-unsloth.yaml --max_notes 256 --stage2_max_notes 256 --batch_sizes 16,32,64,96,128,256`
  - `.venv/bin/python scripts/sweep_single_h100.py --config configs/runs/medgemma-27b-text-it-unsloth.yaml --max_notes 256 --stage2_max_notes 256 --max_new_tokens 256 --stage1_batch_size 128 --stage1_max_num_seqs 128,256 --stage1_max_num_batched_tokens 8192,16384,32768 --stage1_chunked_prefill 1 --batch_sizes 32,64,96,128,256`
- Tests:
  - `uv run open-patients-test`

## Change Guidelines
- If you change schema structure or field names:
  - update schema JSON
  - confirm prompt generation still reflects enums/types
  - run tests (`test_schema_loader*`, `test_prompts`, `test_extraction`)
- If you change config/profile behavior:
  - validate precedence in `tests/test_cli_overrides.py` and `tests/test_config.py`
- If you change output file naming/resume behavior:
  - validate `tests/test_writer.py` and replica metadata aggregation paths
- Keep CLI backward compatibility where possible (`--config` + direct overrides).

## Notes For Future Agents
See `notes/agent-onboarding.md` and `notes/change-checklist.md`.
