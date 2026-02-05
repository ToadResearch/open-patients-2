"""
General utility functions for the enrichment pipeline.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Optional, Any, Dict

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_json_extract(text: str) -> Optional[dict]:
    """
    Extract first JSON object-like substring and parse it.
    Returns dict or None.
    """
    text = text.strip()
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    blob = m.group(0).strip()
    try:
        return json.loads(blob)
    except Exception:
        # Very small, safe-ish repairs: remove trailing commas.
        blob2 = re.sub(r",(\s*[}\]])", r"\1", blob)
        try:
            return json.loads(blob2)
        except Exception:
            return None


def truncate_note(note: str, max_chars: int) -> str:
    """Truncate long notes, keeping head and tail for context."""
    if len(note) <= max_chars:
        return note
    head = note[: max_chars // 2]
    tail = note[-(max_chars // 2) :]
    return head + "\n...\n" + tail


def make_chat_prompt(
    tokenizer: Any,
    system: str,
    user: str,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    force_plain: bool = False,
) -> str:
    """
    Use tokenizer chat template if available; otherwise fall back to a simple format.

    chat_template_kwargs lets us pass model-specific switches like:
      - enable_thinking=False  (Qwen3-style reasoning off)
    Set force_plain=True to bypass chat templates even if available.
    """
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if (not force_plain) and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                **(chat_template_kwargs or {}),
            )
        except TypeError:
            # Older tokenizers may not accept extra kwargs.
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        except Exception:
            pass

    # fallback
    return f"{system}\n\nUser:\n{user}\n\nAssistant:\n"
