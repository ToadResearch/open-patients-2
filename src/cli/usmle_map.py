#!/usr/bin/env python3
"""Create configs/usmle_mapping.json if missing (idempotent)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.utils import colored, print_header

def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(description="Generate USMLE mapping file if missing.")
    ap.add_argument(
        "--output",
        default="configs/usmle_mapping.json",
        help="Output path for mapping JSON (default: configs/usmle_mapping.json)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the output file already exists",
    )
    return ap.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    # Parse only our args, pass the rest through to the mapping tool.
    args, rest = parse_args(argv)
    print_header("USMLE Mapping")

    output_path = Path(args.output)
    if output_path.exists() and output_path.stat().st_size > 0 and not args.force:
        print(
            f"{colored('Skipping:', 'YELLOW')} {colored(str(output_path), 'CYAN')} "
            "already exists."
        )
        return

    from ..utils import usmle_mappings

    # Ensure output is set for the tool.
    if "--output" not in rest:
        rest = ["--output", str(output_path)] + rest

    code = usmle_mappings.main(rest)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
