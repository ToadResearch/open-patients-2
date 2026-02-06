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


if __name__ == "__main__":
    unittest.main()
