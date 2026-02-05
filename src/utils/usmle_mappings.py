#!/usr/bin/env python3
"""Map Open-Patients usmle-* _id rows to MedQA-USMLE us_qbank indices.

Creates a mapping from Open-Patients _id (usmle-<number>) to MedQA index by
matching Open-Patients description to MedQA question. If all shifts
(usmle_id - medqa_index) are identical, emits a top-level `shift` key.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from datasets import load_dataset


def build_medqa_index(
    dataset_name: str,
    split: str,
    index_col: str,
    question_col: str,
    streaming: bool,
) -> Tuple[Dict[str, List[int]], int, int]:
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    qmap: Dict[str, List[int]] = {}
    total = 0
    missing = 0

    for row in ds:
        total += 1
        if question_col not in row or index_col not in row:
            missing += 1
            continue
        q = row[question_col]
        idx = row[index_col]
        if not isinstance(q, str):
            missing += 1
            continue
        try:
            idx_int = int(idx)
        except Exception:  # noqa: BLE001
            missing += 1
            continue
        key = q
        qmap.setdefault(key, []).append(idx_int)

    return qmap, total, missing


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map Open-Patients usmle-* ids to MedQA-USMLE indices."
    )
    parser.add_argument(
        "--medqa-dataset",
        type=str,
        default="mkieffer/MedQA-USMLE",
        help="Hugging Face dataset name for MedQA-USMLE",
    )
    parser.add_argument(
        "--medqa-split",
        type=str,
        default="us_qbank",
        help="MedQA split to use (default: us_qbank)",
    )
    parser.add_argument(
        "--medqa-index-col",
        type=str,
        default="index",
        help="MedQA index column name (default: index)",
    )
    parser.add_argument(
        "--medqa-question-col",
        type=str,
        default="question",
        help="MedQA question column name (default: question)",
    )
    parser.add_argument(
        "--open-dataset",
        type=str,
        default="ncbi/Open-Patients",
        help="Hugging Face dataset name for Open-Patients",
    )
    parser.add_argument(
        "--open-split",
        type=str,
        default="train",
        help="Open-Patients split to use (default: train)",
    )
    parser.add_argument(
        "--open-id-col",
        type=str,
        default="_id",
        help="Open-Patients id column name (default: _id)",
    )
    parser.add_argument(
        "--open-desc-col",
        type=str,
        default="description",
        help="Open-Patients description column name (default: description)",
    )
    parser.add_argument(
        "--streaming-medqa",
        action="store_true",
        help="Stream the MedQA dataset instead of downloading all at once",
    )
    parser.add_argument(
        "--streaming-open",
        action="store_true",
        help="Stream the Open-Patients dataset instead of downloading all at once",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("usmle_mapping.json"),
        help="JSON output path (default: usmle_mapping.json in CWD)",
    )
    parser.add_argument(
        "--max-missing-samples",
        type=int,
        default=5,
        help="Max number of missing samples to include in output (default: 5)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    qmap, medqa_total, medqa_missing = build_medqa_index(
        args.medqa_dataset,
        args.medqa_split,
        args.medqa_index_col,
        args.medqa_question_col,
        args.streaming_medqa,
    )

    open_ds = load_dataset(args.open_dataset, split=args.open_split, streaming=args.streaming_open)

    usmle_pattern = re.compile(r"usmle-(\d+)")

    mapping: Dict[str, int] = {}
    ambiguous: Dict[str, List[int]] = {}
    missing: List[Dict[str, str]] = []
    shifts: Dict[int, int] = {}
    duplicate_pairs: Dict[str, List[str]] = {}
    dropped_duplicate_ids: List[str] = []

    total_open = 0
    usmle_rows = 0
    matched = 0
    missing_count = 0
    ambiguous_count = 0

    # First pass: collect USMLE descriptions to detect duplicates
    desc_to_ids: Dict[str, List[str]] = defaultdict(list)
    open_rows: List[dict] = []
    for row in open_ds:
        total_open += 1
        rec_id = row.get(args.open_id_col)
        if rec_id is None:
            continue
        rec_id_str = str(rec_id)
        m = usmle_pattern.fullmatch(rec_id_str)
        if not m:
            continue
        desc = row.get(args.open_desc_col)
        if isinstance(desc, str):
            desc_to_ids[desc].append(rec_id_str)
        open_rows.append(row)

    for desc, ids in desc_to_ids.items():
        if len(ids) > 1:
            duplicate_pairs[desc] = ids
            dropped_duplicate_ids.extend(ids[1:])

    # Second pass: do mapping, skipping dropped duplicate ids
    for row in open_rows:
        rec_id = row.get(args.open_id_col)
        if rec_id is None:
            continue
        rec_id_str = str(rec_id)
        m = usmle_pattern.fullmatch(rec_id_str)
        if not m:
            continue
        if rec_id_str in dropped_duplicate_ids:
            continue
        usmle_rows += 1

        desc = row.get(args.open_desc_col)
        if not isinstance(desc, str):
            missing_count += 1
            if len(missing) < args.max_missing_samples:
                missing.append({"_id": rec_id_str, "reason": "missing description"})
            continue

        key = desc
        candidates = qmap.get(key)
        if not candidates:
            missing_count += 1
            if len(missing) < args.max_missing_samples:
                missing.append({"_id": rec_id_str, "reason": "no matching question"})
            continue

        if len(candidates) > 1:
            ambiguous_count += 1
            ambiguous[rec_id_str] = candidates
            # still choose the first for mapping/shift to keep deterministic output
        idx = candidates[0]
        mapping[rec_id_str] = idx
        matched += 1

        usmle_num = int(m.group(1))
        shift = usmle_num - idx
        shifts[shift] = shifts.get(shift, 0) + 1

    result: Dict[str, object] = {}
    if len(shifts) == 1 and matched > 0:
        only_shift = next(iter(shifts))
        result["shift"] = only_shift

    result.update(
        {
            "medqa": {
                "dataset": args.medqa_dataset,
                "split": args.medqa_split,
                "total": medqa_total,
                "missing_fields": medqa_missing,
            },
            "open_patients": {
                "dataset": args.open_dataset,
                "split": args.open_split,
                "total": total_open,
                "usmle_rows": usmle_rows,
            },
            "counts": {
                "matched": matched,
                "ambiguous": ambiguous_count,
                "missing": missing_count,
            },
            # "shift_counts": shifts,
            "dropped_duplicate_ids": dropped_duplicate_ids,
            "duplicate_pairs": duplicate_pairs,
            "mapping": mapping,
        }
    )

    if missing:
        result["missing_samples"] = missing
    if ambiguous:
        result["ambiguous_mappings"] = ambiguous

    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is None:
        print(output_json)
    else:
        out_path = args.output
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"Wrote output to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
