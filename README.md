# Open-Patients+

Enrich `ncbi/Open-Patients` into an indexable JSONL dataset using a local vLLM model.

## Quickstart (Step-by-Step)

1. Set up the environment:

```bash
uv venv
source .venv/bin/activate
uv sync
```

On Linux GPU machines, install the vLLM extra:

```bash
uv sync --extra vllm
```

2. Ensure the USMLE mapping file exists:

```bash
uv run open-patients-usmle-map
```

This will skip if `configs/usmle_mapping.json` is already present.

3. Run an enrichment job using a run profile:

```bash
uv run open-patients-worker \
	--config configs/runs/medgemma-27b-text-it.yaml
```

Relative `out_dir` values are automatically placed under `outputs/`.

When `--resume` is not set, outputs are written to a new run subfolder under `out_dir`
(e.g., `outputs/open_patients_enriched_medgemma27b/run_YYYYmmdd_HHMMSS_xxxxxx`).

4. (Optional) Launch multi-GPU replica runs:

```bash
uv run open-patients-replicas \
	--config configs/runs/medgemma-27b-text-it.yaml \
	--gpus 0,1,2,3,4,5,6,7
```

5. Push the enriched dataset to Hugging Face Hub:

```bash
uv run open-patients-push \
	--data_dir outputs/open_patients_enriched_medgemma27b/run_YYYYmmdd_HHMMSS_xxxxxx \
	--repo_name open-patients-enriched-medgemma27b \
	--org your-organization \
	--private
```

**Hugging Face token:** Required for pushing. You can either:
- run `uv run hf auth login` once (uses a cached token), or
- pass `--token YOUR_TOKEN`, or
- set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` in your environment.

## Tests

Run the unit tests:

```bash
uv run open-patients-test
```

## Benchmarking

Run a quick throughput benchmark (defaults to 500 notes):

```bash
uv run open-patients-bench \
	--config configs/runs/medgemma-27b-text-it.yaml
```

Run a multi-GPU replica benchmark (one process per GPU):

```bash
uv run open-patients-bench-replicas \
	--config configs/runs/medgemma-27b-text-it.yaml \
	--gpus 0,1,2,3
```

Note: `--max_notes` applies per replica in the multi-process benchmark.

You can also add a `benchmark:` section to a run profile to set defaults (e.g., `max_notes`, `batch_size`).

By default, metrics are written under `benchmarks/`. Replica runs create a folder with
per-replica metrics plus a `bench_metadata.json` aggregate.
Override the note count or choose a custom metrics path:

```bash
uv run open-patients-bench \
	--config configs/runs/medgemma-27b-text-it.yaml \
	--max_notes 500 \
	--json_out benchmarks/bench_metrics.json
```

## Run Details

See Quickstart above for common commands. This section captures flag/reference details.

Command naming:
- `open-patients-worker` runs a single enrichment process.
- `open-patients-replicas` launches multiple workers across GPUs.
  It also assigns per-replica tags to avoid file collisions and writes a combined
  `run_metadata.json` after all replicas finish.
- `open-patients-bench` runs a single-process throughput benchmark.
- `open-patients-bench-replicas` launches multi-process benchmarks across GPUs and writes
  a `bench_metadata.json` aggregate.

---

## What it produces

For each row, the script outputs a JSON record including:

**Record**
- `id` (original Open-Patients `_id`)

**Provenance**
- `source` (URL back to origin; derived from input `_id`)
- `usmle-*` -> HF viewer row URL
- `pmc-*` -> PMC article URL
- `trec-cds-*` / `trec-ct-*` -> TREC CDS year URL

**Scalars**
- `age_years`
- `sex` (`M|F`)
- `pregnant` (`pregnant|not_pregnant|null`)
- `chief_complaint`
- `primary_diagnosis`
- `care_setting` (`ED|inpatient|outpatient|ICU|unknown|null`)

**Typed Lists (with structured attributes)**

- `conditions`: list of `{name, status, body_site, laterality, certainty, evidence}`
  - Diagnoses/diseases/medical problems (including PMH/comorbidities)
  - `status`: `present|negated|uncertain|historical`
  - `certainty`: `suspected|confirmed|ruled_out|unknown`

- `symptoms_signs`: list of `{name, status, body_site, laterality, certainty, evidence}`
  - Patient-reported symptoms and clinician-observed signs/exam findings (HPI/ROS/PE)

- `medications`: list of `{name, status, dose, unit, route, frequency, evidence}`
  - Individual medication mentions with dosing details
  - `status`: `current|discontinued|historical|prescribed`

- `procedures`: list of `{name, status, body_site, laterality, date, evidence}`
  - Discrete clinical actions/events (tests, biopsies, surgeries, interventions)
  - `status`: `performed|planned|cancelled|recommended`

- `treatments`: list of `{type, name, dose, unit, fractions, boost_dose, boost_unit, route, frequency, cycles, status, evidence}`
  - Therapy courses/regimens (radiation, chemo cycles, endocrine/targeted plans)
  - `type`: `surgery|radiation|chemotherapy|immunotherapy|hormone_therapy|targeted_therapy|medication|other`

- `observations`: list of `{category, test, status, certainty, value, unit, interpretation, flag, body_site, laterality, location_detail, finding, assessment, method, measurements, evidence}`
  - Objective measurements, test results, and scored assessments (imaging, pathology, labs, vitals, clinical scores)
  - `category`: `imaging|pathology|lab|vital|clinical|genomics|microbiology|device`
  - `flag`: `positive|negative|high|low|normal|abnormal|equivocal|unknown` (normalized label for filtering)

- `family_history`: list of `{condition, relative, relative_type, age_at_diagnosis, status, evidence}`
  - Family history of conditions (not patient's own diagnoses)

**Audit**
- `extraction_ok` (bool)
- `created_at` (UTC ISO timestamp)
- if extraction fails: `model_output_raw` is stored for debugging


### Notes on flags

`--config`
Load a run profile YAML (see `configs/runs/*.yaml`). CLI flags override values from the profile.

`--structured_output`
**Recommended.** Uses vLLM's structured output feature with JSON schema constrained decoding (via xgrammar/guidance backends). This guarantees the output conforms to the schema defined in `configs/schemas/schema.json`, eliminating JSON parsing errors.

`--schema`
Path to the JSON schema wrapper (default: `configs/schemas/schema.json`).

`--resume`
Skips input `_id`s already present in `processed_ids.txt` (stored in `out_dir`). Useful for long runs.

`--batch_size`
Increase for throughput if you have GPU headroom. If you OOM, reduce it.

`--max_input_chars`
Truncates long notes by keeping the beginning and end with an ellipsis in the middle.

`--tensor_parallel_size`
Set `>1` if your vLLM is running across multiple GPUs.

`--prompt_style`
Use `compact` to avoid dumping large enum lists into the prompt (recommended with structured output).

`--prompt_mode`
Use `plain` to bypass the tokenizer chat template for non-chat models.

`--chat_template_kwargs`
JSON dict of tokenizer chat-template kwargs (merged with `--disable_thinking`).

`--run_id`
Optional run folder name under `out_dir` (used by `open-patients-replicas`).

`--run_tag`
Prefix for output shard/metadata filenames (useful for multi-process runs).

### Multi-process sharding (N workers)

If you want to run multiple processes (on one machine or across nodes) without overlap, use `--num_shards` and `--shard_idx`.

Example: 4 local workers:

```bash
for i in 0 1 2 3; do
	uv run open-patients-worker \
		--model gpt-oss-20b \
		--out_dir outputs/open_patients_enriched \
		--batch_size 32 \
		--structured_output \
		--resume \
		--num_shards 4 \
		--shard_idx "$i" \
		--run_tag "r$i" &
done
wait
```

Each worker processes a deterministic hash-based partition of the input `_id`.
`--run_tag` ensures each worker writes distinct shard files and metadata.

You can also use the launcher (reads `parallel.replicas` from the run profile):

```bash
uv run open-patients-replicas \
	--config configs/runs/medgemma-27b-text-it.yaml
```

### Output files

Inside `--out_dir` (or the run subfolder when `--resume` is not set):

- `data_shard_00000.jsonl`, `data_shard_00001.jsonl`, ...
	Enriched dataset shards.
- `processed_ids.txt`
	One input `_id` per line; used by `--resume`.
- `run_metadata.json`
	Run metadata (runtime, tokens/sec, config, and counters).

For multi-process runs (with `open-patients-replicas` or manual `--run_tag`):
- `data_shard_r0_00000.jsonl`, `data_shard_r1_00000.jsonl`, ...
- `processed_ids_r0.txt`, `processed_ids_r1.txt`, ...
- `run_metadata_r0.json`, `run_metadata_r1.json`, ...
- `run_metadata.json` (aggregated across replicas; written by `open-patients-replicas`).


## Loading the output with Hugging Face Datasets

```python
from datasets import load_dataset

ds = load_dataset(
		"json",
    data_files="outputs/open_patients_enriched_medgemma27b/run_YYYYmmdd_HHMMSS_xxxxxx/*.jsonl",
		split="train",
)

print(ds.column_names)
print(ds[0])
```

Convert to Parquet for fast filtering:

```python
ds.to_parquet("outputs/open_patients_enriched_medgemma27b/run_YYYYmmdd_HHMMSS_xxxxxx.parquet")
```

## Pushing to Hugging Face Hub

Use the `open-patients-push` command to upload your enriched dataset (see Quickstart for a full example).

### Push options

| Flag | Description |
|------|-------------|
| `--data_dir` | Directory containing the enriched JSONL shards (required) |
| `--repo_name` | Name of the HF dataset repository (required) |
| `--org` | Organization or username (defaults to your personal account) |
| `--private` | Make the repository private |
| `--parquet` | Convert to Parquet before pushing (recommended for large datasets) |
| `--max_shard_size` | Max shard size for Parquet (e.g., `500MB`, `1GB`) |
| `--commit_message` | Custom commit message |
| `--token` | HF API token (uses cached token from `huggingface-cli login` if not provided) |
