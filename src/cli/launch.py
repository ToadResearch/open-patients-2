#!/usr/bin/env python3
"""
Launcher for replica-style sharded runs (one process per shard worker).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List

from ..core.config import load_run_config
from ..utils.utils import colored, print_header


def _combine_all_jsonl_shards(shards_dir: Path, out_path: Path) -> int:
    """Concatenate all *.jsonl files in shards_dir into out_path. Returns shard count."""
    shard_paths = sorted(shards_dir.glob("*.jsonl"))
    if not shard_paths:
        return 0

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as out_f:
        for p in shard_paths:
            with p.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f)
    tmp_path.replace(out_path)
    return len(shard_paths)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Launch replica-sharded Open-Patients runs.")
    ap.add_argument("--config", required=True, help="Run profile YAML (configs/runs/*.yaml)")
    ap.add_argument(
        "--replicas", type=int, default=None, help="Override parallel.replicas from config"
    )
    ap.add_argument("--run_id", default=None, help="Optional run id subfolder name (under out_dir)")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without launching")
    ap.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the enrich script (use -- to separate)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print_header("Open-Patients Replicas")
    cfg = load_run_config(args.config)
    parallel = cfg.get("parallel") or {}
    replicas = args.replicas or parallel.get("replicas") or 1
    if replicas < 1:
        raise SystemExit("replicas must be >= 1")

    extra = list(args.extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    def _find_arg_value(argv: List[str], key: str) -> str | None:
        if key not in argv:
            return None
        idx = argv.index(key)
        if idx + 1 < len(argv):
            return argv[idx + 1]
        return None

    def _has_flag(argv: List[str], flag: str) -> bool:
        return flag in argv

    def _resolve_out_dir(p: Path) -> Path:
        if p.is_absolute():
            return p
        if p.parts and p.parts[0] == "outputs":
            return p
        return Path("outputs") / p

    run_cfg = cfg.get("run") or {}
    base_out_dir_raw = _find_arg_value(extra, "--out_dir") or run_cfg.get("out_dir")
    if base_out_dir_raw:
        base_out_dir = _resolve_out_dir(Path(base_out_dir_raw))
    else:
        base_out_dir = None
    if not base_out_dir:
        raise SystemExit("Missing out_dir (pass --out_dir or set run.out_dir in config).")

    resume = _has_flag(extra, "--resume") or bool(run_cfg.get("resume"))

    run_id = args.run_id
    run_dir = None
    if not resume:
        if not run_id:
            ts = time.strftime("%Y%m%d_%H%M%S")
            suffix = uuid.uuid4().hex[:6]
            run_id = f"run_{ts}_{suffix}"
        run_dir = Path(base_out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    elif run_id:
        run_dir = Path(base_out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[2]
    procs: List[subprocess.Popen] = []
    for i in range(replicas):
        cmd = [
            sys.executable,
            "-m",
            "src.cli.enrich",
            "--config",
            args.config,
            "--num_shards",
            str(replicas),
            "--shard_idx",
            str(i),
            "--run_tag",
            f"r{i}",
        ] + extra
        if run_id:
            cmd += ["--run_id", run_id]

        print(colored(f"[launch] shard {i}/{replicas - 1}: {' '.join(cmd)}", "CYAN"))
        if args.dry_run:
            continue
        procs.append(subprocess.Popen(cmd, cwd=root))

    if args.dry_run:
        return

    exit_code = 0
    for p in procs:
        rc = p.wait()
        if rc != 0:
            exit_code = rc
    if exit_code != 0:
        raise SystemExit(exit_code)

    out_root = run_dir if run_dir is not None else base_out_dir

    replica_meta = []
    for i in range(replicas):
        path = out_root / f"run_metadata_r{i}.json"
        if path.exists():
            try:
                replica_meta.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass

    if replica_meta:

        def _parse_iso(ts: str) -> dt.datetime:
            return dt.datetime.fromisoformat(ts)

        starts = [m.get("start_time") for m in replica_meta if m.get("start_time")]
        ends = [m.get("end_time") for m in replica_meta if m.get("end_time")]
        start_dt = min((_parse_iso(s) for s in starts), default=None)
        end_dt = max((_parse_iso(s) for s in ends), default=None)
        wall_time = (end_dt - start_dt).total_seconds() if start_dt and end_dt else None

        sum_notes = sum(m.get("notes_written", 0) for m in replica_meta)
        sum_input = sum(m.get("input_tokens", 0) for m in replica_meta)
        sum_output = sum(m.get("output_tokens", 0) for m in replica_meta)
        sum_total = sum_input + sum_output
        sum_gen = sum(m.get("gen_time_s", 0.0) for m in replica_meta)

        def _safe_div(num: float, den: float | None) -> float:
            return num / den if den else 0.0

        aggregate = {
            "run_id": run_id,
            "base_out_dir": str(base_out_dir),
            "out_dir": str(out_root),
            "replicas": replicas,
            "resume": bool(resume),
            "config_path": args.config,
            "config": cfg,
            "start_time": start_dt.isoformat() if start_dt else None,
            "end_time": end_dt.isoformat() if end_dt else None,
            "wall_time_s": wall_time,
            "gen_time_s": sum_gen,
            "notes_written": sum_notes,
            "input_tokens": sum_input,
            "output_tokens": sum_output,
            "total_tokens": sum_total,
            "notes_per_s": _safe_div(sum_notes, wall_time),
            "total_toks_per_s": _safe_div(sum_total, wall_time),
            "input_toks_per_s": _safe_div(sum_input, sum_gen),
            "output_toks_per_s": _safe_div(sum_output, sum_gen),
            "replica_metadata_files": [
                str(out_root / f"run_metadata_r{i}.json") for i in range(replicas)
            ],
        }
        (out_root / "run_metadata.json").write_text(
            json.dumps(aggregate, indent=2),
            encoding="utf-8",
        )

    shards_dir = out_root / "shards"
    if shards_dir.exists():
        combined_path = out_root / "data.jsonl"
        n_shards = _combine_all_jsonl_shards(shards_dir, combined_path)
        if n_shards:
            print(colored(f"[combine] wrote {combined_path} from {n_shards} shard(s)", "CYAN"))


if __name__ == "__main__":
    main()
