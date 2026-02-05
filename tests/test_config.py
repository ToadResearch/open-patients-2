import tempfile
import unittest
from pathlib import Path

from src.core.config import config_to_defaults, load_run_config


class ConfigTests(unittest.TestCase):
    def test_config_to_defaults_sampling_alias(self) -> None:
        cfg = {
            "run": {"dataset": "ncbi/Open-Patients", "structured_output": True},
            "generation": {"temperature": 0.1},
            "prompt": {"chat_template_kwargs": {"thinking_mode": "off"}},
        }
        defaults = config_to_defaults(cfg)
        self.assertEqual(defaults["dataset"], "ncbi/Open-Patients")
        self.assertEqual(defaults["temperature"], 0.1)
        self.assertTrue(defaults["structured_output"])
        self.assertEqual(defaults["chat_template_kwargs"], {"thinking_mode": "off"})

    def test_load_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text("run:\n  dataset: test", encoding="utf-8")
            cfg = load_run_config(str(path))
            self.assertEqual(cfg["run"]["dataset"], "test")


if __name__ == "__main__":
    unittest.main()
