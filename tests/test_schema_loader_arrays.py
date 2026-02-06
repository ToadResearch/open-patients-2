import json
import tempfile
import unittest
from pathlib import Path

from src.core.schema_loader import load_schema


class SchemaLoaderArrayTests(unittest.TestCase):
    def test_array_scalar_vs_object_fields(self) -> None:
        wrapper = {
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "description": "Age"},
                        "conditions": {
                            "type": "array",
                            "description": "Condition names",
                            "items": {"type": "string", "enum": ["a", "b"]},
                        },
                        "medications": {
                            "type": "array",
                            "description": "Medication objects",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            },
                        },
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema.json"
            path.write_text(json.dumps(wrapper), encoding="utf-8")
            bundle = load_schema(path)

        self.assertIn("age", bundle.scalar_names)
        self.assertIn("conditions", bundle.typed_list_names)
        self.assertIn("medications", bundle.typed_list_names)

        array_scalar_names = {f["name"] for f in bundle.array_scalar_fields}
        array_object_names = {f["name"] for f in bundle.array_object_fields}

        self.assertIn("conditions", array_scalar_names)
        self.assertIn("medications", array_object_names)

        # array_scalar_fields should carry enum metadata
        conditions_def = next(f for f in bundle.array_scalar_fields if f["name"] == "conditions")
        self.assertEqual(conditions_def["item_type"], "string")
        self.assertEqual(conditions_def["allowed_values"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
