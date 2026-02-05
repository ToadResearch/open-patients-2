import unittest
from pathlib import Path

from src.core.extraction import derive_source_url, ensure_schema, load_usmle_id_to_row
from src.core.schema_loader import load_schema


class ExtractionTests(unittest.TestCase):
    def test_derive_source_url(self) -> None:
        mapping = {"usmle-12892": 14366}
        self.assertEqual(
            derive_source_url("usmle-12892", mapping),
            "https://huggingface.co/datasets/mkieffer/MedQA-USMLE/viewer/default/us_qbank?row=14366",
        )
        self.assertEqual(
            derive_source_url("pmc-8696182-3", mapping),
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC8696182/",
        )
        self.assertEqual(
            derive_source_url("trec-cds-2014-8", mapping),
            "https://www.trec-cds.org/2014.html",
        )
        self.assertEqual(
            derive_source_url("trec-ct-2021-10", mapping),
            "https://www.trec-cds.org/2021.html",
        )
        self.assertIsNone(derive_source_url("unknown-1", mapping))

    def test_ensure_schema_defaults(self) -> None:
        schema_path = Path("configs/schemas/schema.json")
        bundle = load_schema(schema_path)

        obj = {
            "age": 50,
            "conditions": "not-a-list",
        }
        out = ensure_schema(obj, bundle.schema_keys, bundle.typed_list_names)

        self.assertIn("age", out)
        self.assertIn("conditions", out)
        self.assertEqual(out["age"], 50)
        self.assertEqual(out["conditions"], [])

        # Missing keys should be added.
        self.assertIn("sex", out)

    def test_load_usmle_mapping(self) -> None:
        mapping_path = Path("configs/usmle_mapping.json")
        mapping = load_usmle_id_to_row(mapping_path)
        self.assertIsInstance(mapping, dict)
        if mapping:
            k = next(iter(mapping.keys()))
            self.assertEqual(k, k.lower())


if __name__ == "__main__":
    unittest.main()
