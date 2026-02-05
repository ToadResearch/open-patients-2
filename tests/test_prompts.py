import unittest
from pathlib import Path

from src.core.prompts import build_system_prompt
from src.core.schema_loader import load_schema


class PromptTests(unittest.TestCase):
    def test_compact_prompt_avoids_large_enums(self) -> None:
        schema_path = Path("configs/schemas/schema.json")
        bundle = load_schema(schema_path)

        compact = build_system_prompt(bundle, style="compact")
        verbose = build_system_prompt(bundle, style="verbose")

        # Compact prompt should not inline enum values.
        self.assertNotIn("aerospace_medicine", compact)
        self.assertIn("aerospace_medicine", verbose)

        # Compact prompt should still describe allowed sets.
        self.assertIn("allowed set", compact)


if __name__ == "__main__":
    unittest.main()
