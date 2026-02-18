import tempfile
import unittest
from pathlib import Path

from src.core.config import config_to_defaults, load_run_config


class ConfigTests(unittest.TestCase):
    def test_config_to_defaults_api_mapping(self) -> None:
        cfg = {
            "run": {
                "dataset": "ncbi/Open-Patients",
                "structured_output": True,
                "samples": 11,
            },
            "generation": {"temperature": 0.1},
            "api": {
                "timeout_s": 30.0,
                "max_retries": 2,
                "retry_backoff_initial_s": 0.5,
                "retry_backoff_max_s": 3.0,
                "outage_abort_after_s": 120.0,
                "endpoints": [
                    {
                        "name": "local",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "model": "foo/bar",
                    }
                ],
            },
            "prompt": {
                "chat_template_kwargs": {"thinking_mode": "off"},
                "schema_in_prompt": True,
            },
            "vllm": {"tensor_parallel_size": 8},
        }
        defaults = config_to_defaults(cfg)
        self.assertEqual(defaults["dataset"], "ncbi/Open-Patients")
        self.assertEqual(defaults["temperature"], 0.1)
        self.assertTrue(defaults["structured_output"])
        self.assertEqual(defaults["max_notes"], 11)
        self.assertEqual(defaults["api_timeout_s"], 30.0)
        self.assertEqual(defaults["api_max_retries"], 2)
        self.assertEqual(defaults["api_retry_backoff_initial_s"], 0.5)
        self.assertEqual(defaults["api_retry_backoff_max_s"], 3.0)
        self.assertEqual(defaults["api_outage_abort_after_s"], 120.0)
        self.assertEqual(defaults["api_endpoints"][0]["name"], "local")
        self.assertEqual(defaults["chat_template_kwargs"], {"thinking_mode": "off"})
        self.assertTrue(defaults["schema_in_prompt"])
        self.assertNotIn("tensor_parallel_size", defaults)

    def test_run_samples_negative_means_all(self) -> None:
        cfg = {"run": {"samples": -1}}
        defaults = config_to_defaults(cfg)
        self.assertEqual(defaults["max_notes"], 0)

    def test_load_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text("run:\n  dataset: test", encoding="utf-8")
            cfg = load_run_config(str(path))
            self.assertEqual(cfg["run"]["dataset"], "test")


if __name__ == "__main__":
    unittest.main()
