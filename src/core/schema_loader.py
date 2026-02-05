"""
JSON structured output schema loader for clinical note extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class SchemaBundle:
    schema_path: Path
    wrapper: Dict[str, Any]
    json_schema: Dict[str, Any]
    properties: Dict[str, Any]
    scalar_names: List[str]
    typed_list_names: List[str]
    schema_keys: List[str]
    typed_list_fields: List[Dict[str, Any]]
    scalar_fields: List[Dict[str, Any]]
    array_object_fields: List[Dict[str, Any]]
    array_scalar_fields: List[Dict[str, Any]]


def load_schema(schema_path: Path) -> SchemaBundle:
    """Load a JSON Schema wrapper from disk and derive prompt/extraction helpers."""
    with schema_path.open("r", encoding="utf-8") as f:
        wrapper: Dict[str, Any] = json.load(f)

    json_schema: Dict[str, Any] = wrapper["json_schema"]["schema"]
    properties: Dict[str, Any] = json_schema["properties"]

    scalar_names: List[str] = []
    typed_list_names: List[str] = []
    for field_name, field_def in properties.items():
        field_type = field_def.get("type")
        if field_type == "array":
            typed_list_names.append(field_name)
        else:
            scalar_names.append(field_name)

    schema_keys: List[str] = scalar_names + typed_list_names

    typed_list_fields: List[Dict[str, Any]] = []
    array_object_fields: List[Dict[str, Any]] = []
    array_scalar_fields: List[Dict[str, Any]] = []

    for field_name in typed_list_names:
        field_def = properties[field_name]
        items = field_def.get("items", {}) or {}
        item_type = items.get("type")
        item_props = items.get("properties", {}) if isinstance(items, dict) else {}

        typed_list_fields.append(
            {
                "name": field_name,
                "description": field_def.get("description", ""),
                "item_schema": item_props if item_props else {},
            }
        )

        if item_type == "object" or item_props:
            array_object_fields.append(
                {
                    "name": field_name,
                    "description": field_def.get("description", ""),
                    "item_schema": item_props,
                }
            )
        else:
            examples = items.get("examples") if isinstance(items, dict) else None
            array_scalar_fields.append(
                {
                    "name": field_name,
                    "description": field_def.get("description", ""),
                    "item_type": item_type or "string",
                    "allowed_values": items.get("enum") if isinstance(items, dict) else None,
                    "example": examples[0] if examples else None,
                }
            )

    scalar_fields: List[Dict[str, Any]] = []
    for field_name in scalar_names:
        field_def = properties[field_name]
        scalar_fields.append(
            {
                "name": field_name,
                "description": field_def.get("description", ""),
                "type": field_def.get("type"),
                "allowed_values": field_def.get("enum"),
                "example": field_def.get("examples", [None])[0]
                if field_def.get("examples")
                else None,
            }
        )

    return SchemaBundle(
        schema_path=schema_path,
        wrapper=wrapper,
        json_schema=json_schema,
        properties=properties,
        scalar_names=scalar_names,
        typed_list_names=typed_list_names,
        schema_keys=schema_keys,
        typed_list_fields=typed_list_fields,
        scalar_fields=scalar_fields,
        array_object_fields=array_object_fields,
        array_scalar_fields=array_scalar_fields,
    )
