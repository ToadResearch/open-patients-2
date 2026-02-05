import json
import tempfile
import unittest
from pathlib import Path

from src.core.writer import JSONLShardedWriter, ProcessedIdWriter, load_processed_ids


class WriterTests(unittest.TestCase):
    def test_jsonl_sharded_writer_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            writer = JSONLShardedWriter(out_dir=out_dir, shard_size=1, name_prefix="data_shard_r0")
            writer.write({"id": "a"})
            writer.write({"id": "b"})
            writer.close()

            f0 = out_dir / "data_shard_r0_00000.jsonl"
            f1 = out_dir / "data_shard_r0_00001.jsonl"
            self.assertTrue(f0.exists())
            self.assertTrue(f1.exists())

            self.assertEqual(json.loads(f0.read_text().strip()), {"id": "a"})
            self.assertEqual(json.loads(f1.read_text().strip()), {"id": "b"})

    def test_processed_ids_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "processed_ids.txt"
            writer = ProcessedIdWriter(path, flush_every=1)
            writer.add("a")
            writer.add("b")
            writer.close()

            loaded = load_processed_ids(path)
            self.assertEqual(loaded, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
