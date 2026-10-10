"""Strict JSON extraction from model output.

Small open-weight models rarely emit clean JSON. They wrap it in prose, fence
it, trail a comma, or emit a trailing explanation. This module turns that
mess into either a dict or a clear failure, and it *records why* it failed so
the eval suite can report a real malformed-output rate rather than a guess.

The rule throughout: never guess. If we cannot parse, we return None and the
caller falls back to the deterministic generator.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Fenced block, optionally tagged ```json
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> tuple[dict[str, Any] | None, str]:
    """Pull the first JSON object out of `text`.

    Returns ``(obj, reason)``. ``obj`` is None when nothing parseable was
    found; ``reason`` is a short machine-friendly failure tag.

    Strategy, in order of reliability:
      1. the whole string, stripped
      2. a fenced ```json block
      3. the outermost brace-balanced span

    Balance-aware scanning (step 3) matters: naive ``text[text.find('{'):]``
    truncates at a brace inside a string value, which is exactly what happens
    when a model writes "look for a \"blue\" sign".
    """
    if not text or not text.strip():
        return None, "empty_output"

    raw = text.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, "ok"
    except json.JSONDecodeError:
        pass

    fenced = _FENCE_RE.search(text)
    if fenced:
        try:
            obj = json.loads(fenced.group(1).strip())
            if isinstance(obj, dict):
                return obj, "ok_fenced"
        except json.JSONDecodeError:
            pass

    span = _outermost_object(text)
    if span:
        try:
            obj = json.loads(span)
            if isinstance(obj, dict):
                return obj, "ok_extracted"
        except json.JSONDecodeError:
            pass

    return None, "unparseable_json"


def _outermost_object(text: str) -> str | None:
    """Return the first balanced {...} span, respecting strings and escapes."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = in_string
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def coerce_list(value: Any, *, limit: int, max_len: int = 120) -> list[str]:
    """Normalise a model-supplied list of strings.

    Small models return a bare string, a comma-joined string, or a list
    containing numbers. This flattens all of those into a clean list, which is
    what keeps one sloppy field from failing an otherwise good generation.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    else:
        parts = [str(value).strip()]

    return [p[:max_len] for p in parts if p][:limit]


def coerce_str(value: Any, *, max_len: int, default: str = "") -> str:
    """Normalise a model-supplied string, rejecting nested structures."""
    if isinstance(value, str):
        return value.strip()[:max_len]
    if isinstance(value, (int, float, bool)):
        return str(value)[:max_len]
    return default


def coerce_int(value: Any, *, lo: int, hi: int, default: int) -> int:
    """Coerce to int within bounds, tolerating "20 minutes" and floats."""
    if isinstance(value, bool):
        return default
    try:
        if isinstance(value, str):
            digits = re.sub(r"[^0-9-]", "", value)
            if not digits:
                return default
            n = int(digits)
        elif isinstance(value, (int, float)):
            n = int(value)
        else:
            return default
    except (ValueError, TypeError):
        return default
    return max(lo, min(hi, n))


def coerce_float(value: Any, *, lo: float, hi: float, default: float) -> float:
    """Coerce to float within bounds.

    Model confidence arrives as `0.85`, `"0.85"`, `"85%"`, `1`, or sometimes as
    the string `"high"`. Anything unrecognised falls back to `default` rather
    than to 0.0 or 1.0 -- guessing an extreme here would distort every number
    the eval suite reports.
    """
    if isinstance(value, bool):
        return default

    if isinstance(value, str):
        text = value.strip().lower()
        named = {"high": hi, "medium": (lo + hi) / 2, "moderate": (lo + hi) / 2, "low": lo}
        if text in named:
            return named[text]
        text = text.rstrip("%")
        try:
            value = float(text)
        except ValueError:
            return default

    if isinstance(value, (int, float)):
        n = float(value)
        # A bare 0-1 vs 0-100 scale mix-up is common; normalise it.
        if hi <= 1.0 and n > 1.0:
            n = n / 100.0
        if not (n == n):  # NaN check without importing math
            return default
        return max(lo, min(hi, n))

    return default