import unittest

from src.utils.utils import safe_json_extract, split_reasoning_and_final


class SafeJsonExtractTests(unittest.TestCase):
    def test_extracts_plain_object(self) -> None:
        self.assertEqual(safe_json_extract('{"a": 1}'), {"a": 1})

    def test_extracts_with_prefix_suffix(self) -> None:
        text = "noise before\n{\"a\": 1, \"b\": 2}\nnoise after"
        self.assertEqual(safe_json_extract(text), {"a": 1, "b": 2})

    def test_handles_extra_closing_brace_after_valid_object(self) -> None:
        text = "{\"a\": 1}\n}"
        self.assertEqual(safe_json_extract(text), {"a": 1})

    def test_trailing_comma_repair(self) -> None:
        text = "{\"a\": 1,}"
        self.assertEqual(safe_json_extract(text), {"a": 1})


class SplitReasoningTests(unittest.TestCase):
    def test_split_think_tag(self) -> None:
        reasoning, final = split_reasoning_and_final("<think>hidden steps</think>{\"a\":1}")
        self.assertEqual(reasoning, "hidden steps")
        self.assertEqual(final, "{\"a\":1}")

    def test_split_multiple_reasoning_blocks(self) -> None:
        reasoning, final = split_reasoning_and_final(
            "<think>a</think><reasoning>b</reasoning>{\"x\":2}"
        )
        self.assertEqual(reasoning, "a\n\nb")
        self.assertEqual(final, "{\"x\":2}")

    def test_no_reasoning_tags(self) -> None:
        reasoning, final = split_reasoning_and_final("{\"k\":3}")
        self.assertIsNone(reasoning)
        self.assertEqual(final, "{\"k\":3}")


if __name__ == "__main__":
    unittest.main()
