import tempfile
import unittest
from pathlib import Path

from src.cli.usmle_map import main


class UsmleMapTests(unittest.TestCase):
    def test_skip_when_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usmle_mapping.json"
            path.write_text("sentinel", encoding="utf-8")
            # Should skip and leave file untouched.
            main(["--output", str(path)])
            self.assertEqual(path.read_text(encoding="utf-8"), "sentinel")


if __name__ == "__main__":
    unittest.main()
