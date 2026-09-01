"""Patient-facing text. No string a patient reads is written in Python."""
from __future__ import annotations

import json
from functools import lru_cache

from .config import STRINGS_DIR, settings


@lru_cache(maxsize=8)
def _bundle(lang: str) -> dict[str, str]:
    path = STRINGS_DIR / f"{lang}.json"
    if not path.exists():
        path = STRINGS_DIR / "en.json"
    return json.loads(path.read_text(encoding="utf-8"))


def t(key: str, lang: str | None = None, **kwargs: object) -> str:
    """Look up a string and fill its placeholders."""
    text = _bundle(lang or settings.lang).get(key)
    if text is None:
        return f"[{key}]"
    return text.format(**kwargs) if kwargs else text
