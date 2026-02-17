#!/usr/bin/env python3
"""
Sweep Open-Patients vLLM/worker knobs for single-GPU throughput.

This script runs `open-patients-bench` repeatedly and ranks configs by `total_toks_per_s`.

Why a script?
- Some CLI flags are `store_true` and cannot override a config default of `true` back to `false`.
  For those booleans we generate temporary YAML run profiles per sweep point.

Typical usage (single H100):
  .venv/bin/python scripts/sweep_single_h100.py --config configs/runs/medgemma-4b-it-unsloth.yaml
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def _csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _csv_bools01(s: str) -> List[bool]:
    out: List[bool] = []
    for x in s.split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x in {"1", "true", "t", "yes", "y", "on"}:
            out.append(True)
        elif x in {"0", "false", "f", "no", "n", "off"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"Invalid bool value: {x!r}")
    return out


def _bench_exe() -> str:
    local = Path(".venv") / "bin" / "open-patients-bench"
    if local.exists():
        return str(local)
    return "open-patients-bench"


def _deep_set(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur: Dict[str, Any] = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def _safe_name(parts: Dict[str, Any]) -> str:
    # Stable-ish identifier for filenames.
    items = sorted(parts.items(), key=lambda kv: kv[0])
    s = "_".join(f"{k}={v}" for k, v in items)
    s = s.replace("/", "_").replace(" ", "")
    for ch in [":", ",", "{", "}", "[", "]", "(", ")", "'", '"']:
        s = s.replace(ch, "")
    return s


def _run_bench(
    *,
    cfg: Dict[str, Any],
    cfg_path: Path,
    json_out: Path,
    batch_size: int,
    max_notes: int,
    max_new_tokens: int,
    extra_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cmd = [
        _bench_exe(),
        "--config",
        str(cfg_path),
        "--batch_size",
        str(batch_size),
        "--max_notes",
        str(max_notes),
        "--max_new_tokens",
        str(max_new_tokens),
        "--json_out",
        str(json_out),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    except Exception as exc:
        return None, f"failed to spawn bench: {exc}"

    if proc.returncode != 0:
        tail = (proc.stdout or "")[-4000:]
        return None, f"bench exited {proc.returncode}\n{tail}"

    try:
        metrics = json.loads(json_out.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"bench succeeded but failed to read metrics JSON: {exc}"

    # Attach the effective vllm settings we swept (for easier downstream diffing).
    metrics["_sweep_cfg"] = {
        "vllm": (cfg.get("vllm") or {}),
        "model": (cfg.get("model") or {}),
    }
    return metrics, None


def _print_top(rows: List[Dict[str, Any]], n: int = 10) -> None:
    rows = sorted(rows, key=lambda r: float(r.get("total_toks_per_s") or 0.0), reverse=True)
    print("\nTop results (by total_toks_per_s):")
    print("  rank  total_toks/s   batch  max_seqs  max_batched_toks  chunked_prefill")
    for i, r in enumerate(rows[:n], 1):
        vllm = (r.get("_sweep_cfg") or {}).get("vllm") or {}
        print(
            f"  {i:>4}  {float(r.get('total_toks_per_s') or 0.0):>11.1f}"
            f"  {int(r.get('batch_size') or 0):>6}"
            f"  {int(vllm.get('max_num_seqs') or 0):>8}"
            f"  {int(vllm.get('max_num_batched_tokens') or 0):>16}"
            f"  {str(bool(vllm.get('enable_chunked_prefill'))):>15}"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep single-GPU bench configs and rank throughput.")
    ap.add_argument("--config", required=True, help="Base run profile YAML (configs/runs/*.yaml)")
    ap.add_argument("--max_notes", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--out_dir", default=None, help="Output directory under benchmarks/")

    # Stage 1: engine-level knobs (rebuild vLLM engine each point).
    ap.add_argument("--stage1_batch_size", type=int, default=32)
    ap.add_argument("--stage1_max_num_seqs", type=_csv_ints, default="128,256")
    ap.add_argument(
        "--stage1_max_num_batched_tokens",
        type=_csv_ints,
        default="16384,32768,65536,131072",
    )
    ap.add_argument("--stage1_chunked_prefill", type=_csv_bools01, default="0,1")
    ap.add_argument("--stage1_prefix_caching", type=_csv_bools01, default="1")
    ap.add_argument("--stage1_gpu_memory_utilization", type=_csv_floats, default="0.92")

    # Stage 2: request-level knobs on the best stage1 engine config.
    ap.add_argument("--batch_sizes", type=_csv_ints, default="8,16,32,48,64,96,128,192,256")
    ap.add_argument(
        "--stage2_max_notes",
        type=int,
        default=None,
        help="Override max_notes for stage 2 (default: --max_notes).",
    )

    ap.add_argument("--no_stage2", action="store_true", help="Skip batch size sweep stage")
    ap.add_argument("--keep_configs", action="store_true", help="Keep generated YAML configs")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    # Make progress visible even when stdout is piped (common in remote runners).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args(argv)

    base_cfg_path = Path(args.config)
    if not base_cfg_path.exists():
        raise SystemExit(f"Config not found: {base_cfg_path}")

    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base_cfg, dict):
        raise SystemExit(f"Config must be a YAML mapping: {base_cfg_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    out_dir = Path(args.out_dir) if args.out_dir else Path("benchmarks") / f"sweep_{ts}_{suffix}"
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dir = out_dir / "configs"
    metrics_dir = out_dir / "metrics"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base config: {base_cfg_path}")
    print(f"Output dir:  {out_dir}")

    stage1_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    stage1_grid = list(
        itertools.product(
            args.stage1_max_num_seqs,
            args.stage1_max_num_batched_tokens,
            args.stage1_chunked_prefill,
            args.stage1_prefix_caching,
            args.stage1_gpu_memory_utilization,
        )
    )
    print(f"\nStage 1 points: {len(stage1_grid)}")

    for (max_seqs, max_batched_toks, chunked, prefix_cache, mem_util) in stage1_grid:
        if args.stage1_batch_size > max_seqs:
            continue

        cfg = copy.deepcopy(base_cfg)
        _deep_set(cfg, ("vllm", "max_num_seqs"), int(max_seqs))
        _deep_set(cfg, ("vllm", "max_num_batched_tokens"), int(max_batched_toks))
        _deep_set(cfg, ("vllm", "enable_chunked_prefill"), bool(chunked))
        _deep_set(cfg, ("vllm", "enable_prefix_caching"), bool(prefix_cache))
        _deep_set(cfg, ("vllm", "gpu_memory_utilization"), float(mem_util))

        point = {
            "max_num_seqs": int(max_seqs),
            "max_num_batched_tokens": int(max_batched_toks),
            "enable_chunked_prefill": bool(chunked),
            "enable_prefix_caching": bool(prefix_cache),
            "gpu_memory_utilization": float(mem_util),
            "batch_size": int(args.stage1_batch_size),
        }

        cfg_name = _safe_name(point)
        cfg_path = cfg_dir / f"stage1_{cfg_name}.yaml"
        json_out = metrics_dir / f"stage1_{cfg_name}.json"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        print(
            f"[stage1] batch={args.stage1_batch_size} max_seqs={max_seqs}"
            f" max_batched_toks={max_batched_toks} chunked={chunked} prefix_cache={prefix_cache}"
            f" mem={mem_util}"
        )
        metrics, err = _run_bench(
            cfg=cfg,
            cfg_path=cfg_path,
            json_out=json_out,
            batch_size=args.stage1_batch_size,
            max_notes=args.max_notes,
            max_new_tokens=args.max_new_tokens,
        )
        if err:
            failures.append({"point": point, "error": err})
            print(f"  -> FAIL: {err.splitlines()[0]}")
            continue

        assert metrics is not None
        stage1_rows.append(metrics)
        print(f"  -> total_toks_per_s={metrics.get('total_toks_per_s')}")

    if not stage1_rows:
        raise SystemExit("No successful stage1 runs. See failures in output dir.")

    stage1_best = max(stage1_rows, key=lambda r: float(r.get("total_toks_per_s") or 0.0))
    print("\nStage 1 best:")
    _print_top([stage1_best], n=1)

    best_cfg_vllm = (stage1_best.get("_sweep_cfg") or {}).get("vllm") or {}
    best_engine_cfg = copy.deepcopy(base_cfg)
    for k in [
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "gpu_memory_utilization",
    ]:
        if k in best_cfg_vllm:
            _deep_set(best_engine_cfg, ("vllm", k), best_cfg_vllm[k])

    stage2_rows: List[Dict[str, Any]] = []
    if not args.no_stage2:
        max_seqs = int(best_cfg_vllm.get("max_num_seqs") or 0)
        stage2_max_notes = int(args.stage2_max_notes) if args.stage2_max_notes else int(args.max_notes)
        batch_sizes = [bs for bs in args.batch_sizes if (bs <= max_seqs and bs <= stage2_max_notes)]
        skipped = [bs for bs in args.batch_sizes if (bs <= max_seqs and bs > stage2_max_notes)]
        if skipped:
            print(
                f"\n[warn] Skipping stage 2 batch_sizes that exceed stage2_max_notes={stage2_max_notes}: "
                f"{skipped}"
            )
        print(
            f"\nStage 2 batch sizes (<= max_num_seqs={max_seqs}, <= stage2_max_notes={stage2_max_notes}): "
            f"{batch_sizes}"
        )
        for bs in batch_sizes:
            point = {"batch_size": int(bs)}
            cfg_name = _safe_name(point)
            cfg_path = cfg_dir / f"stage2_{cfg_name}.yaml"
            json_out = metrics_dir / f"stage2_{cfg_name}.json"
            cfg_path.write_text(yaml.safe_dump(best_engine_cfg, sort_keys=False), encoding="utf-8")

            print(f"[stage2] batch={bs}")
            metrics, err = _run_bench(
                cfg=best_engine_cfg,
                cfg_path=cfg_path,
                json_out=json_out,
                batch_size=bs,
                max_notes=stage2_max_notes,
                max_new_tokens=args.max_new_tokens,
            )
            if err:
                failures.append({"point": {"stage2_batch_size": bs}, "error": err})
                print(f"  -> FAIL: {err.splitlines()[0]}")
                continue

            assert metrics is not None
            stage2_rows.append(metrics)
            print(f"  -> total_toks_per_s={metrics.get('total_toks_per_s')}")

    all_rows = stage1_rows + stage2_rows
    best = max(all_rows, key=lambda r: float(r.get("total_toks_per_s") or 0.0))

    print("\nOverall best:")
    _print_top([best], n=1)
    _print_top(all_rows, n=10)

    summary = {
        "base_config": str(base_cfg_path),
        "out_dir": str(out_dir),
        "args": vars(args),
        "best": best,
        "stage1_best": stage1_best,
        "rows": all_rows,
        "failures": failures,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {out_dir / 'summary.json'}")

    if not args.keep_configs:
        # Keep only the summary + metrics; configs can be regenerated and are noisy.
        for p in cfg_dir.glob("*.yaml"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            cfg_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()
