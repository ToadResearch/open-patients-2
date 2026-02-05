"""
JSONL sharded writer and resume tracking utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List


class JSONLShardedWriter:
    """Write records to sharded JSONL files."""

    def __init__(
        self,
        out_dir: Path,
        shard_size: int = 50_000,
        name_prefix: str = "data_shard",
    ) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.name_prefix = name_prefix
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_idx = 0
        self.count_in_shard = 0
        self.f = None
        self._open_new()

    def _open_new(self) -> None:
        if self.f:
            self.f.close()
        fname = self.out_dir / f"{self.name_prefix}_{self.shard_idx:05d}.jsonl"
        self.f = fname.open("a", encoding="utf-8")
        self.count_in_shard = 0
        self.shard_idx += 1

    def write(self, rec: dict) -> None:
        assert self.f is not None
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.count_in_shard += 1
        if self.count_in_shard >= self.shard_size:
            self._open_new()

    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None


def load_processed_ids(processed_path: Path) -> set:
    """Load set of already-processed IDs from resume file."""
    if not processed_path.exists():
        return set()
    out = set()
    with processed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(line)
    return out


class ProcessedIdWriter:
    """
    Keep processed_ids.txt open and append in buffered batches.

    This avoids per-record open/close overhead, which becomes significant at 100k+ rows.
    """

    def __init__(self, processed_path: Path, flush_every: int = 2000) -> None:
        self.processed_path = processed_path
        self.flush_every = flush_every
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.processed_path.open("a", encoding="utf-8")
        self._buf: List[str] = []

    def add(self, _id: str) -> None:
        self._buf.append(_id)
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        self._f.write("\n".join(self._buf) + "\n")
        self._buf.clear()
        self._f.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._f.close()
