import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.cli import bench
from src.cli import enrich
from src.cli import launch
from src.cli import serve_vllm
from src.core.llm_api import EndpointConfig


class _FakeProc:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.returncode = None

    def wait(self):
        self.returncode = 0
        return 0


class LauncherTests(unittest.TestCase):
    def test_worker_delegates_to_replicas_launcher(self) -> None:
        args = SimpleNamespace(
            config="configs/runs/openrouter-trinity-mini.yaml",
            replicas=4,
            num_shards=1,
            shard_idx=0,
            run_tag=None,
            run_id=None,
        )
        calls = []

        def fake_call(cmd, cwd=None):
            calls.append((cmd, cwd))
            return 0

        with patch("src.cli.enrich.subprocess.call", side_effect=fake_call), patch(
            "src.cli.enrich.sys.argv",
            [
                "op-worker",
                "--config",
                "configs/runs/openrouter-trinity-mini.yaml",
                "--replicas",
                "4",
                "--max_notes",
                "5",
                "--queue_size",
                "10",
                "--run_id",
                "run_test",
            ],
        ):
            with self.assertRaises(SystemExit) as ctx:
                enrich._delegate_to_replicas(args)

        self.assertEqual(ctx.exception.code, 0)
        self.assertTrue(calls)
        cmd = calls[0][0]
        self.assertEqual(cmd[1:3], ["-m", "src.cli.launch"])
        self.assertIn("--replicas", cmd)
        self.assertIn("4", cmd)
        self.assertIn("--max_notes", cmd)
        self.assertIn("--queue_size", cmd)
        self.assertNotIn("--run_id", cmd)

    def test_bench_delegates_to_bench_replicas_launcher(self) -> None:
        args = SimpleNamespace(
            config="configs/runs/openrouter-trinity-mini.yaml",
            replicas=3,
            num_shards=1,
            shard_idx=0,
            run_tag=None,
            json_out=None,
        )
        calls = []

        def fake_call(cmd, cwd=None):
            calls.append((cmd, cwd))
            return 0

        with patch("src.cli.bench.subprocess.call", side_effect=fake_call), patch(
            "src.cli.bench.sys.argv",
            [
                "op-bench",
                "--config",
                "configs/runs/openrouter-trinity-mini.yaml",
                "--replicas",
                "3",
                "--max_notes",
                "9",
                "--queue_size",
                "6",
            ],
        ):
            with self.assertRaises(SystemExit) as ctx:
                bench._delegate_to_bench_replicas(args)

        self.assertEqual(ctx.exception.code, 0)
        self.assertTrue(calls)
        cmd = calls[0][0]
        self.assertEqual(cmd[1:3], ["-m", "src.cli.bench_replicas"])
        self.assertIn("--replicas", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--max_notes", cmd)
        self.assertIn("--queue_size", cmd)

    def test_replicas_launcher_does_not_set_cuda_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            out_dir = Path(tmp) / "out"
            cfg_path.write_text(
                f"""
run:
  out_dir: {out_dir}
  resume: true
parallel:
  replicas: 1
""".strip(),
                encoding="utf-8",
            )

            popen_kwargs = []

            def fake_popen(*args, **kwargs):
                popen_kwargs.append(kwargs)
                return _FakeProc(*args, **kwargs)

            with patch("src.cli.launch.subprocess.Popen", side_effect=fake_popen), patch(
                "sys.argv",
                ["op-replicas", "--config", str(cfg_path), "--replicas", "1"],
            ):
                launch.main()

            self.assertTrue(popen_kwargs)
            self.assertNotIn("env", popen_kwargs[0])

    def test_vllm_serve_command_generation(self) -> None:
        endpoint = EndpointConfig(
            name="ep",
            base_url="http://127.0.0.1:8123/v1",
            model="foo/model",
            serve={
                "host": "0.0.0.0",
                "port": 8123,
                "cuda_visible_devices": "2,3",
                "tensor_parallel_size": 2,
                "enable_chunked_prefill": True,
                "max_model_len": 4096,
                "args": ["--served-model-name", "foo-model"],
            },
        )
        cmd, env = serve_vllm._build_vllm_cmd(endpoint)

        self.assertEqual(cmd[:3], ["vllm", "serve", "foo/model"])
        self.assertIn("--host", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("--tensor-parallel-size", cmd)
        self.assertIn("--enable-chunked-prefill", cmd)
        self.assertIn("--max-model-len", cmd)
        self.assertEqual(env.get("CUDA_VISIBLE_DEVICES"), "2,3")


if __name__ == "__main__":
    unittest.main()
