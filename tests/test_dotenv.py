import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src


class DotenvTests(unittest.TestCase):
    def test_load_dotenv_from_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OP_DOTENV_TEST=from_file\n", encoding="utf-8")
            prev_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                with patch.dict(os.environ, {}, clear=True):
                    src._load_dotenv()
                    self.assertEqual(os.environ.get("OP_DOTENV_TEST"), "from_file")
            finally:
                os.chdir(prev_cwd)

    def test_load_dotenv_does_not_override_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OP_DOTENV_TEST=from_file\n", encoding="utf-8")
            prev_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                with patch.dict(os.environ, {"OP_DOTENV_TEST": "from_env"}, clear=True):
                    src._load_dotenv()
                    self.assertEqual(os.environ.get("OP_DOTENV_TEST"), "from_env")
            finally:
                os.chdir(prev_cwd)


if __name__ == "__main__":
    unittest.main()
