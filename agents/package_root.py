"""Helpers for deriving importable Python package roots from repository slugs."""

from __future__ import annotations

import keyword
import re


def normalize_python_package_root(name: str, *, default: str = "src") -> str:
    """Return a valid top-level Python package name for a repo/layout slug."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or "").strip())
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    if not text:
        text = default
    if text and text[0].isdigit():
        text = f"pkg_{text}"
    if keyword.iskeyword(text):
        text = f"{text}_pkg"
    return text
