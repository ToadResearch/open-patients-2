import unittest
from pathlib import Path

from src.core.schema_loader import load_schema


class SchemaLoaderTests(unittest.TestCase):
    def test_load_schema_bundle(self) -> None:
        schema_path = Path("configs/schemas/schema.json")
        bundle = load_schema(schema_path)

        self.assertTrue(bundle.schema_keys)
        self.assertIn("age", bundle.scalar_names)
        self.assertIn("conditions", bundle.typed_list_names)

        # Arrays of scalars vs objects are classified correctly.
        scalar_arrays = {f["name"] for f in bundle.array_scalar_fields}
        object_arrays = {f["name"] for f in bundle.array_object_fields}
        self.assertIn("specialties", scalar_arrays)
        self.assertIn("conditions", object_arrays)

        self.assertIn("json_schema", bundle.wrapper)
        self.assertIn("properties", bundle.json_schema)


if __name__ == "__main__":
    unittest.main()
