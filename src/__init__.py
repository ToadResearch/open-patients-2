from __future__ import annotations

from pathlib import Path


def _load_dotenv() -> None:
    """
    Load a local .env for CLI runs without requiring manual `export`.

    Existing shell env vars win (override=False), so explicit exports still take precedence.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return

    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
        return

    # Fallback to repo root when cwd isn't the project directory.
    repo_env = Path(__file__).resolve().parents[1] / ".env"
    if repo_env.exists():
        load_dotenv(repo_env, override=False)


_load_dotenv()

__all__ = ["__version__"]
__version__ = "0.1.0"
