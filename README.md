# Open-Patients+

Enrich `ncbi/Open-Patients` into structured JSONL using OpenAI-compatible Chat Completions endpoints.

The worker supports:
- one or many endpoints (`api.endpoints`)
- dynamic multi-endpoint request scheduling
- schema-oriented JSON extraction
- sharded outputs + resume tracking
- failed-record artifacts

## Quickstart

1. Set up environment:

```bash
uv venv
source .venv/bin/activate
uv sync
```

API keys can be stored in a local `.env` file (for example `GEMINI_API_KEY=...`).
CLI commands auto-load `.env`, so manual `export` is not required.

If you want to run local vLLM servers, install vLLM extra:

```bash
uv sync --extra vllm
```

2. Ensure USMLE mapping exists:

```bash
uv run op-usmle-map
```

3. Configure endpoint(s) in a run profile (`configs/runs/*.yaml`) under `api.endpoints`.
   Ready-to-run provider presets:
   - `configs/runs/openrouter-trinity-mini.yaml`
   - `configs/runs/gemini-3-flash-preview.yaml`
   - Gemini preset includes `extra_body.reasoning_effort: minimal` to reduce structured-output truncation.

4. (Optional) Start local vLLM server(s) from config:

```bash
uv run op-vllm-serve --config configs/runs/qwen3-4b-thinking-2507-fp8-vllm.yaml
```

5. Run enrichment:

```bash
uv run op-worker --config configs/runs/qwen3-4b-thinking-2507-fp8-vllm.yaml
```

6. (Optional) Launch replica-sharded client workers from the main worker command:

```bash
uv run op-worker --config configs/runs/medgemma-27b-text-it-unsloth.yaml --replicas 8
```

7. Benchmark throughput:

```bash
uv run op-bench --config configs/runs/medgemma-27b-text-it-unsloth.yaml
```

8. (Optional) Run replica-sharded benchmark workers from the same benchmark command:

```bash
uv run op-bench --config configs/runs/medgemma-27b-text-it-unsloth.yaml --replicas 8
```

## Run Profile Shape (API)

```yaml
run:
  dataset: ncbi/Open-Patients
  split: train
  out_dir: ./open_patients_medgemma27b_unsloth
  resume: true
  # -1 (or omitted) => full dataset; positive integer => sample limit
  samples: -1
  shard_size: 50000
  schema: configs/schemas/schema.json
  usmle_mapping: configs/usmle_mapping.json

model:
  name: unsloth/medgemma-27b-text-it
  prompt_mode: chat

api:
  timeout_s: 120.0
  max_retries: 4
  retry_backoff_initial_s: 1.0
  retry_backoff_max_s: 30.0
  outage_abort_after_s: 900.0
  endpoints:
    - name: medgemma27b
      base_url: http://127.0.0.1:8000/v1
      model: unsloth/medgemma-27b-text-it
      api_key_env: OPENAI_API_KEY
      concurrency: 128
      structured_mode: json_schema # json_schema | json_object | none
      extra_body: {}
      serve:
        host: 127.0.0.1
        port: 8000
        cuda_visible_devices: "0"
        tensor_parallel_size: 1
        dtype: auto
        max_model_len: 8192
        gpu_memory_utilization: 0.92
        enable_chunked_prefill: true
        enable_prefix_caching: true
        kv_cache_dtype: fp8
        calculate_kv_scales: false
        max_num_batched_tokens: 8192
        max_num_seqs: 128
        max_parallel_loading_workers: 2

sampling:
  temperature: 0.0
  top_p: 0.95
  max_new_tokens: 8192
  seed: 0
  structured_output: true

parallel:
  replicas: 8
```

## Multi-Endpoint Scheduling

`op-worker` uses a dynamic async queue:
- each endpoint has its own concurrency (`api.endpoints[].concurrency`)
- requests are pulled by whichever endpoint worker is free
- faster endpoints naturally process more notes

## Structured Output

- `sampling.structured_output: true` enables provider structured response mode where possible.
- `prompt.schema_in_prompt: true` embeds full schema text in prompt and disables structured response mode for that run.
- parsing fallback still uses JSON extraction, and failed parses are retained.

## Failure Behavior

Per-record failures are written, not dropped:
- record in main dataset with `extraction_ok: false`
- raw model text and error metadata retained

Additional artifacts:
- `failed_ids*.txt`
- `shards/failed_records*.jsonl`

Run-level outage behavior:
- worker retries per-request with exponential backoff
- if all endpoints stay unhealthy past `api.outage_abort_after_s`, the run aborts

## Output Files

Inside `out_dir` (or run subfolder when `--resume` is false):

- `shards/data_shard_00000.jsonl`, `shards/data_shard_00001.jsonl`, ...
- `data.jsonl` (single-process auto-combine; replica runs combine after all shards finish)
- `processed_ids*.txt`
- `failed_ids*.txt` (only when failures occur)
- `shards/failed_records*.jsonl` (only when failures occur)
- `run_metadata*.json`

## Commands

Primary commands:
- `op-worker`: Main enrichment run. Usage: `uv run op-worker --config <run.yaml> [--replicas N]`.
- `op-bench`: Throughput benchmark run. Usage: `uv run op-bench --config <run.yaml> [--replicas N]`.
- `op-vllm-serve`: Starts one or more local `vllm serve` API servers from `api.endpoints[].serve`.
- `op-check-prompt`: Renders a sample prompt exactly as the model sees it.
- `op-prompt-stats`: Computes prompt token/char length distribution.
- `op-prompt-stats-replicas`: Parallel prompt stats across shard replicas with merged summary.
- `op-usmle-map`: Creates `configs/usmle_mapping.json` if missing (or regenerates with `--force`).
- `op-push`: Uploads generated dataset outputs to Hugging Face Hub.
- `op-test`: Runs the unit test suite.
- `op-view`: Opens the local output viewer for browsing generated records.

Compatibility wrappers (optional):
- `op-replicas`: Explicit replica launcher wrapper (equivalent to worker replica mode).
- `op-bench-replicas`: Explicit benchmark replica launcher wrapper (equivalent to bench replica mode).

## Tests

Run unit tests:

```bash
uv run op-test
```
