import unittest
from pathlib import Path

from src.core.prompts import build_system_prompt
from src.core.schema_loader import load_schema


class PromptTests(unittest.TestCase):
    def test_prompt_includes_enum_values(self) -> None:
        schema_path = Path("configs/schemas/schema.json")
        bundle = load_schema(schema_path)

        prompt = build_system_prompt(bundle)

        # Enums should be included in the prompt.
        self.assertIn("aerospace_medicine", prompt)

    def test_prompt_can_embed_full_schema_wrapper(self) -> None:
        schema_path = Path("configs/schemas/schema.json")
        bundle = load_schema(schema_path)

        prompt = build_system_prompt(bundle, include_json_schema=True)

        self.assertIn("JSON schema wrapper (authoritative):", prompt)
        self.assertIn('"type": "json_schema"', prompt)


if __name__ == "__main__":
    unittest.main()
