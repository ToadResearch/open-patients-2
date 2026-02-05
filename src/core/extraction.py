"""
Schema validation and source derivation for clinical note extraction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Dict, Sequence


def ensure_schema(obj: dict, schema_keys: Sequence[str], typed_list_names: Sequence[str]) -> dict:
    """
    Ensure all required keys exist. Drop extras.
    """
    out = {}
    for k in schema_keys:
        out[k] = obj.get(k, None)
    # Normalize all typed list fields (object lists)
    for lk in typed_list_names:
        v = out.get(lk)
        if v is None:
            out[lk] = []
        elif not isinstance(v, list):
            out[lk] = []
    return out


def load_usmle_id_to_row(mapping_path: Path) -> Dict[str, int]:
    """
    Load configs/usmle_mapping.json and return a dict:
      {"usmle-0": 0, "usmle-1": 1, ...}
    Keys are normalized to lowercase.
    """
    with mapping_path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    m = blob.get("mapping", {}) or {}
    out: Dict[str, int] = {}
    for k, v in m.items():
        try:
            out[str(k).lower()] = int(v)
        except Exception:
            continue
    return out


_USMLE_RE = re.compile(r"^usmle-(\d+)$", re.IGNORECASE)
_PMC_RE = re.compile(r"^pmc-(\d+)-\d+$", re.IGNORECASE)
_TREC_RE = re.compile(r"^trec-(?:cds|ct)-(\d{4})-\d+$", re.IGNORECASE)


def derive_source_url(_id: str, usmle_id_to_row: Optional[Dict[str, int]] = None) -> Optional[str]:
    """
    Map Open-Patients _id conventions to a canonical source URL.

    - usmle-<num> => HF viewer row via configs/usmle_mapping.json
      https://huggingface.co/datasets/mkieffer/MedQA-USMLE/viewer/default/us_qbank?row=<mapped_int>

    - pmc-<pmc_id>-<num> => PMC article
      https://pmc.ncbi.nlm.nih.gov/articles/PMC<pmc_id>/

    - trec-cds-<year>-<num> => TREC CDS year page
      https://www.trec-cds.org/<year>.html

    - trec-ct-<year>-<num> => same TREC CDS year page (per your rule)
      https://www.trec-cds.org/<year>.html
    """
    s = (_id or "").strip().lower()
    if not s:
        return None

    m = _USMLE_RE.match(s)
    if m:
        if usmle_id_to_row is None:
            return None
        row = usmle_id_to_row.get(s)
        if row is None:
            return None
        return (
            "https://huggingface.co/datasets/mkieffer/MedQA-USMLE/viewer/default/us_qbank"
            f"?row={row}"
        )

    m = _PMC_RE.match(s)
    if m:
        pmc_id = m.group(1)
        return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/"

    m = _TREC_RE.match(s)
    if m:
        year = m.group(1)
        return f"https://www.trec-cds.org/{year}.html"

    return None
