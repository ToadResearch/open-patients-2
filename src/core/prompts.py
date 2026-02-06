"""
Prompt templates for clinical note extraction.

Prompts are generated dynamically from schema.json (JSON Schema format).
"""

from __future__ import annotations

from .schema_loader import SchemaBundle


def _format_type(type_val) -> str:
    """Format a JSON Schema type for display."""
    if isinstance(type_val, list):
        return "|".join(str(t) for t in type_val)
    return str(type_val)


def _build_field_rules(schema: SchemaBundle) -> str:
    """Build the field-specific rules section of the system prompt."""
    lines = []

    # Scalar rules
    for f in schema.scalar_fields:
        allowed = f.get("allowed_values")
        if allowed:
            lines.append(f'- For "{f["name"]}": one of {allowed} or null.')
        # else: no special rule needed beyond "null if unknown"

    # Array-of-objects rules
    for f in schema.array_object_fields:
        item_schema = f.get("item_schema", {})
        if item_schema:
            fields_desc_parts = []
            for k, v in item_schema.items():
                type_str = _format_type(v.get("type", "string"))
                enum_vals = v.get("enum")
                if enum_vals:
                    fields_desc_parts.append(f'"{k}": {type_str} (one of {enum_vals})')
                else:
                    fields_desc_parts.append(f'"{k}": {type_str}')
            fields_desc = ", ".join(fields_desc_parts)
            lines.append(f'- "{f["name"]}" is a list of objects: {{{fields_desc}}}')
        else:
            lines.append(f'- "{f["name"]}" is a list of objects')

    # Array-of-scalars rules
    for f in schema.array_scalar_fields:
        item_type = _format_type(f.get("item_type", "string"))
        enum_vals = f.get("allowed_values")
        if enum_vals:
            lines.append(
                f'- "{f["name"]}" is a list of {item_type} values (one of {enum_vals}).'
            )
        else:
            lines.append(f'- "{f["name"]}" is a list of {item_type} values.')

    return "\n".join(lines)


def build_system_prompt(schema: SchemaBundle) -> str:
    """Build the system prompt."""
    return f"""You extract structured fields from a clinical note.

Return ONE valid JSON object and NOTHING ELSE.

Rules:
- Use null if not present/unknown for scalar fields.
- Use [] (empty list) if none for list fields.
- Do not guess. Only extract what is explicitly supported by the note.
- Keep strings concise. No long sentences.
- Limit each list to at most 30 items (choose the most clinically relevant).
{_build_field_rules(schema)}
"""


USER_TEMPLATE = """Extract the fields from this clinical note.

NOTE:
{note}

JSON KEYS (must match exactly):
{keys}
"""
