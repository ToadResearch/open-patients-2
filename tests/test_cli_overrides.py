import tempfile
import unittest
from pathlib import Path

from src.cli import bench as bench_cli
from src.cli import enrich as enrich_cli


class CliOverrideTests(unittest.TestCase):
    def test_enrich_cli_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text(
                """
run:
  dataset: foo/bar
  out_dir: ./from_config
  samples: 10
model:
  name: test-model
api:
  timeout_s: 33
  max_retries: 2
  endpoints:
    - name: local
      base_url: http://127.0.0.1:8000/v1
      model: test-model
sampling:
  max_new_tokens: 50
  structured_output: true
prompt:
  schema_in_prompt: true
""".strip(),
                encoding="utf-8",
            )

            args = enrich_cli.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--api_max_retries",
                    "5",
                    "--max_new_tokens",
                    "99",
                    "--out_dir",
                    "./from_cli",
                    "--queue_size",
                    "64",
                ]
            )

            self.assertEqual(args.dataset, "foo/bar")
            self.assertEqual(args.model, "test-model")
            self.assertEqual(args.api_timeout_s, 33)
            self.assertEqual(args.api_max_retries, 5)
            self.assertEqual(args.max_new_tokens, 99)
            self.assertEqual(args.out_dir, "./from_cli")
            self.assertEqual(args.queue_size, 64)
            self.assertEqual(args.max_notes, 10)
            self.assertTrue(args.structured_output)
            self.assertTrue(args.schema_in_prompt)
            self.assertFalse(hasattr(args, "tensor_parallel_size"))

    def test_bench_cli_overrides_and_benchmark_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text(
                """
run:
  dataset: foo/bar
  max_notes: 1000
model:
  name: test-model
api:
  endpoints:
    - name: local
      base_url: http://127.0.0.1:8000/v1
      model: test-model
sampling:
  max_new_tokens: 75
benchmark:
  max_notes: 250
  batch_size: 16
""".strip(),
                encoding="utf-8",
            )

            # No CLI overrides: benchmark.max_notes should win over run.max_notes
            args_default = bench_cli.parse_args(["--config", str(cfg_path)])
            self.assertEqual(args_default.max_notes, 250)
            self.assertEqual(args_default.batch_size, 16)
            self.assertEqual(args_default.max_new_tokens, 75)

            # CLI overrides should win over config
            args_override = bench_cli.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--max_notes",
                    "12",
                    "--batch_size",
                    "8",
                    "--queue_size",
                    "20",
                    "--max_new_tokens",
                    "99",
                ]
            )
            self.assertEqual(args_override.max_notes, 12)
            self.assertEqual(args_override.batch_size, 8)
            self.assertEqual(args_override.queue_size, 20)
            self.assertEqual(args_override.max_new_tokens, 99)

    def test_enrich_samples_minus_one_maps_to_all_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text(
                """
run:
  dataset: foo/bar
  out_dir: ./from_config
  samples: -1
model:
  name: test-model
api:
  endpoints:
    - name: local
      base_url: http://127.0.0.1:8000/v1
      model: test-model
""".strip(),
                encoding="utf-8",
            )
            args = enrich_cli.parse_args(["--config", str(cfg_path)])
            self.assertEqual(args.max_notes, 0)


if __name__ == "__main__":
    unittest.main()
