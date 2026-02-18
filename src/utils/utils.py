"""
General utility functions for the enrichment pipeline.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Optional, Any, Dict, Tuple

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_REASONING_TAG_RE = re.compile(
    r"<(?P<tag>think|reasoning)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)


def colored(st, color: str | None, background: bool = False):
    colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    if color is None:
        return st
    return (
        f"\u001b[{10*background+60*(color.upper() == color)+30+colors.index(color.lower())}m"
        f"{st}\u001b[0m"
    )


def print_header(title: str, color: str = "CYAN") -> None:
    line = "-" * len(title)
    print(colored(title, color))
    print(colored(line, color))


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_json_extract(text: str) -> Optional[dict]:
    """
    Extract first JSON object-like substring and parse it.
    Returns dict or None.
    """
    text = text.strip()

    # Primary path: scan for the first parseable JSON object and decode it.
    # raw_decode tolerates trailing content after the parsed object.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    # Fallback: recover object-like span and apply a minimal trailing-comma fix.
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    blob = m.group(0).strip()
    blob2 = re.sub(r",(\s*[}\]])", r"\1", blob)
    try:
        parsed = json.loads(blob2)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def split_reasoning_and_final(text: str) -> Tuple[Optional[str], str]:
    """
    Split assistant output into (reasoning, final_text) when reasoning tags are present.

    For reasoning-enabled open models (commonly via vLLM), outputs often include:
      <think> ... </think>
    or:
      <reasoning> ... </reasoning>
    """
    raw = (text or "").strip()
    if not raw:
        return None, ""

    chunks = []

    def _collect(match: re.Match[str]) -> str:
        body = (match.group("body") or "").strip()
        if body:
            chunks.append(body)
        return ""

    final = _REASONING_TAG_RE.sub(_collect, raw).strip()
    if not chunks:
        return None, raw

    reasoning = "\n\n".join(chunks).strip()
    if not final:
        final = raw
    return (reasoning or None), final


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
