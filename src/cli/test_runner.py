#!/usr/bin/env python3
"""Run the project's unit tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
