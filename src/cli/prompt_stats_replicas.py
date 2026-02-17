#!/usr/bin/env python3
"""
Parallel launcher for prompt length stats (CPU shard replicas + aggregate output).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any

from ..core.config import load_run_config
from ..utils.utils import colored, print_header


def _percentile_nearest_rank(sorted_vals: List[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    n = len(sorted_vals)
    rank = int(math.ceil((pct / 100.0) * n))
    rank = min(max(rank, 1), n)
    return sorted_vals[rank - 1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run open-patients-prompt-stats across multiple shard replicas and aggregate."
    )
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer name")
    ap.add_argument("--config", default=None, help="Run profile YAML (configs/runs/*.yaml)")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel shard workers")
    ap.add_argument("--run_id", default=None, help="Optional benchmark folder name")
    ap.add_argument(
        "--poll_interval",
        type=float,
        default=1.0,
        help="Seconds between progress refreshes",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print commands without launching")
    ap.add_argument(
        "--json_out",
        default=None,
        help="Aggregate stats output path (default: benchmarks/prompts/<run_id>/prompt_stats.json)",
    )
    ap.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to prompt_stats (use -- to separate)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print_header("Open-Patients Prompt Stats Replicas")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    if args.config:
        # Validate config path early for fast failure.
        load_run_config(args.config)

    extra = list(args.extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    managed_flags = {
        "--json_out",
        "--emit_lengths",
        "--no_progress",
        "--quiet",
        "--progress_path",
        "--num_shards",
        "--shard_idx",
    }
    if any(flag in extra for flag in managed_flags):
        raise SystemExit(
            "Do not pass managed shard flags in extra args; "
            "open-patients-prompt-stats-replicas manages those automatically."
        )

    run_id = args.run_id
    if not run_id:
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        run_id = f"prompt_stats_{ts}_{suffix}"

    run_dir = Path("benchmarks/prompts") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]

    procs: List[subprocess.Popen] = []
    log_handles = []
    shard_jsons: List[Path] = []
    progress_jsons: List[Path] = []
    shard_logs: List[Path] = []
    for i in range(args.workers):
        shard_json = run_dir / f"prompt_stats_r{i}.json"
        progress_json = run_dir / f"progress_r{i}.json"
        shard_log = run_dir / f"prompt_stats_r{i}.log"
        shard_jsons.append(shard_json)
        progress_jsons.append(progress_json)
        shard_logs.append(shard_log)

        cmd = [
            sys.executable,
            "-m",
            "src.cli.prompt_stats",
            "--tokenizer",
            args.tokenizer,
            "--num_shards",
            str(args.workers),
            "--shard_idx",
            str(i),
            "--json_out",
            str(shard_json),
            "--emit_lengths",
            "--no_progress",
            "--quiet",
            "--progress_path",
            str(progress_json),
        ]
        if args.config:
            cmd += ["--config", args.config]
        cmd += extra

        print(colored(f"[launch] shard {i}/{args.workers - 1}: {' '.join(cmd)}", "CYAN"))
        if args.dry_run:
            continue
        log_handle = shard_log.open("w", encoding="utf-8")
        log_handles.append(log_handle)
        procs.append(
            subprocess.Popen(
                cmd,
                cwd=root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    if args.dry_run:
        return

    def _read_progress(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"notes_processed": 0, "done": False}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"notes_processed": 0, "done": False}

    line_count = 0

    def _render_progress() -> None:
        nonlocal line_count
        rows = []
        total = 0
        done = 0
        for idx, (proc, path) in enumerate(zip(procs, progress_jsons)):
            prog = _read_progress(path)
            processed = int(prog.get("notes_processed", 0) or 0)
            total += processed
            is_done = bool(prog.get("done", False)) or (proc.poll() is not None and proc.returncode == 0)
            if is_done:
                done += 1
            status = "done" if is_done else "running"
            rows.append(f"shard r{idx}: {processed} ({status})")

        lines = [f"Shard Progress ({done}/{args.workers} done)"]
        lines.extend(rows)
        lines.append(f"total processed: {total}")

        if line_count == 0:
            for line in lines:
                print(line)
        else:
            sys.stdout.write("\x1b[F" * line_count)
            for line in lines:
                sys.stdout.write("\x1b[2K" + line + "\n")
            sys.stdout.flush()
        line_count = len(lines)

    while True:
        _render_progress()
        if all(p.poll() is not None for p in procs):
            break
        time.sleep(max(args.poll_interval, 0.1))

    for h in log_handles:
        h.close()

    exit_code = 0
    failed = []
    for i, p in enumerate(procs):
        rc = p.returncode
        if rc and rc != 0:
            exit_code = rc
            failed.append((i, shard_logs[i], rc))
    if failed:
        print("")
        for idx, log_path, rc in failed:
            print(colored(f"[error] shard r{idx} failed with rc={rc}: {log_path}", "RED"))
        raise SystemExit(exit_code)

    token_lengths: List[int] = []
    char_lengths: List[int] = []
    per_shard = []
    for path in shard_jsons:
        if not path.exists():
            raise SystemExit(f"Missing shard output: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        token_lengths.extend(payload.get("token_lengths", []))
        char_lengths.extend(payload.get("char_lengths", []))
        per_shard.append(
            {
                "file": str(path),
                "notes_processed": payload.get("notes_processed", 0),
                "duration_s": payload.get("duration_s", 0.0),
            }
        )

    if not token_lengths:
        raise SystemExit("No token lengths collected from replicas.")

    tok_sorted = sorted(token_lengths)
    ch_sorted = sorted(char_lengths)

    def _parse_iso(ts: str) -> dt.datetime:
        return dt.datetime.fromisoformat(ts)

    # Find wall time bounds from shard outputs if available.
    shard_payloads = [json.loads(p.read_text(encoding="utf-8")) for p in shard_jsons]
    starts = [p.get("start_time") for p in shard_payloads if p.get("start_time")]
    ends = [p.get("end_time") for p in shard_payloads if p.get("end_time")]
    start_dt = min((_parse_iso(s) for s in starts), default=None)
    end_dt = max((_parse_iso(s) for s in ends), default=None)
    wall_time = (end_dt - start_dt).total_seconds() if start_dt and end_dt else None
    if wall_time is None:
        wall_time = sum(p.get("duration_s", 0.0) for p in shard_payloads)

    aggregate = {
        "tokenizer": args.tokenizer,
        "config": args.config,
        "workers": args.workers,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "notes_processed": len(tok_sorted),
        "wall_time_s": wall_time,
        "notes_per_s": (len(tok_sorted) / wall_time) if wall_time else 0.0,
        "tokens": {
            "min": tok_sorted[0],
            "p50": _percentile_nearest_rank(tok_sorted, 50.0),
            "p95": _percentile_nearest_rank(tok_sorted, 95.0),
            "p99": _percentile_nearest_rank(tok_sorted, 99.0),
            "max": tok_sorted[-1],
            "mean": sum(tok_sorted) / len(tok_sorted),
        },
        "chars": {
            "min": ch_sorted[0],
            "p50": _percentile_nearest_rank(ch_sorted, 50.0),
            "p95": _percentile_nearest_rank(ch_sorted, 95.0),
            "p99": _percentile_nearest_rank(ch_sorted, 99.0),
            "max": ch_sorted[-1],
            "mean": sum(ch_sorted) / len(ch_sorted),
        },
        "per_shard": per_shard,
        "shard_files": [str(p) for p in shard_jsons],
    }

    out_path = Path(args.json_out) if args.json_out else (run_dir / "prompt_stats.json")
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = Path("benchmarks/prompts") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("")
    print(colored("Aggregate Token Length Summary", "CYAN"))
    print(f"  notes: {colored(str(aggregate['notes_processed']), 'GREEN')}")
    print(f"  p50:   {colored(str(aggregate['tokens']['p50']), 'GREEN')}")
    print(f"  p95:   {colored(str(aggregate['tokens']['p95']), 'GREEN')}")
    print(f"  p99:   {colored(str(aggregate['tokens']['p99']), 'GREEN')}")
    print(f"  max:   {colored(str(aggregate['tokens']['max']), 'GREEN')}")
    mean_str = f"{aggregate['tokens']['mean']:.2f}"
    print(f"  mean:  {colored(mean_str, 'GREEN')}")
    print(f"\n{colored('Wrote aggregate:', 'CYAN')} {colored(str(out_path), 'CYAN')}")


if __name__ == "__main__":
    main()
