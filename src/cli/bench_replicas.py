#!/usr/bin/env python3
"""
Launcher for replica-style benchmark runs (one process per GPU).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List

from ..core.config import load_run_config


def _split_csv(val: str) -> List[str]:
    return [x.strip() for x in val.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Launch replica-sharded benchmark runs.")
    ap.add_argument("--config", required=True, help="Run profile YAML (configs/runs/*.yaml)")
    ap.add_argument(
        "--replicas", type=int, default=None, help="Override parallel.replicas from config"
    )
    ap.add_argument("--run_id", default=None, help="Optional run id subfolder name")
    ap.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU IDs (defaults to 0..replicas-1)",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print commands without launching")
    ap.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the bench script (use -- to separate)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_run_config(args.config)
    parallel = cfg.get("parallel") or {}
    replicas = args.replicas or parallel.get("replicas") or 1
    if replicas < 1:
        raise SystemExit("replicas must be >= 1")

    if args.gpus:
        gpus = _split_csv(args.gpus)
    else:
        gpus = [str(i) for i in range(replicas)]

    if len(gpus) < replicas:
        raise SystemExit(f"Not enough GPU IDs for replicas={replicas}: {gpus}")

    extra = list(args.extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if "--json_out" in extra:
        raise SystemExit(
            "open-patients-bench-replicas manages per-replica --json_out paths; "
            "remove --json_out from extra args."
        )

    run_id = args.run_id
    if not run_id:
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        run_id = f"bench_{ts}_{suffix}"

    run_dir = Path("benchmarks") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[2]
    procs: List[subprocess.Popen] = []
    for i in range(replicas):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[i]
        env["PYTHONPATH"] = str(root) + (
            os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else ""
        )

        json_out = run_dir / f"bench_r{i}.json"
        cmd = [
            sys.executable,
            "-m",
            "src.cli.bench",
            "--config",
            args.config,
            "--num_shards",
            str(replicas),
            "--shard_idx",
            str(i),
            "--run_tag",
            f"r{i}",
            "--json_out",
            str(json_out),
        ] + extra

        print(f"[bench] shard {i}/{replicas - 1} on GPU {gpus[i]}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        procs.append(subprocess.Popen(cmd, cwd=root, env=env))

    if args.dry_run:
        return

    exit_code = 0
    for p in procs:
        rc = p.wait()
        if rc != 0:
            exit_code = rc
    if exit_code != 0:
        raise SystemExit(exit_code)

    replica_metrics = []
    for i in range(replicas):
        path = run_dir / f"bench_r{i}.json"
        if path.exists():
            try:
                replica_metrics.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass

    if replica_metrics:
        def _parse_iso(ts: str) -> dt.datetime:
            return dt.datetime.fromisoformat(ts)

        starts = [m.get("start_time") for m in replica_metrics if m.get("start_time")]
        ends = [m.get("end_time") for m in replica_metrics if m.get("end_time")]
        start_dt = min((_parse_iso(s) for s in starts), default=None)
        end_dt = max((_parse_iso(s) for s in ends), default=None)
        wall_time = (end_dt - start_dt).total_seconds() if start_dt and end_dt else None
        if wall_time is None:
            wall_time = max((m.get("total_time_s", 0.0) for m in replica_metrics), default=0.0)

        sum_notes = sum(m.get("notes", 0) for m in replica_metrics)
        sum_batches = sum(m.get("batches", 0) for m in replica_metrics)
        sum_input = sum(m.get("input_tokens", 0) for m in replica_metrics)
        sum_output = sum(m.get("output_tokens", 0) for m in replica_metrics)
        sum_total = sum_input + sum_output
        sum_gen = sum(m.get("gen_time_s", 0.0) for m in replica_metrics)

        def _safe_div(num: float, den: float | None) -> float:
            return num / den if den else 0.0

        aggregate = {
            "run_id": run_id,
            "out_dir": str(run_dir),
            "replicas": replicas,
            "config_path": args.config,
            "config": cfg,
            "start_time": start_dt.isoformat() if start_dt else None,
            "end_time": end_dt.isoformat() if end_dt else None,
            "wall_time_s": wall_time,
            "gen_time_s": sum_gen,
            "notes": sum_notes,
            "batches": sum_batches,
            "input_tokens": sum_input,
            "output_tokens": sum_output,
            "total_tokens": sum_total,
            "notes_per_s": _safe_div(sum_notes, wall_time),
            "total_toks_per_s": _safe_div(sum_total, wall_time),
            "input_toks_per_s": _safe_div(sum_input, sum_gen),
            "output_toks_per_s": _safe_div(sum_output, sum_gen),
            "replica_metrics_files": [
                str(run_dir / f"bench_r{i}.json") for i in range(replicas)
            ],
        }
        (run_dir / "bench_metadata.json").write_text(
            json.dumps(aggregate, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
