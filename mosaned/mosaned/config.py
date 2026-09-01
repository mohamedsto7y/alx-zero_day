"""Settings. Everything the prototype needs to change lives here or in the env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
CLINICAL_DIR = PKG_DIR / "clinical"
FLOWS_DIR = CLINICAL_DIR / "flows"
STRINGS_DIR = PKG_DIR / "strings"
SEEDS_DIR = PKG_DIR / "seeds"


@dataclass(frozen=True)
class Settings:
    # Which LLM backend. One line to change; nothing above the provider layer
    # knows or cares which one is running.
    provider: str = os.getenv("MOSANED_PROVIDER", "stub")
    model: str = os.getenv("MOSANED_MODEL", "qwen2.5:7b-instruct")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    lang: str = os.getenv("MOSANED_LANG", "en")
    db_path: str = os.getenv("MOSANED_DB", str(Path.cwd() / "mosaned.db"))

    # Hard stop so a stuck conversation can't loop forever.
    max_turns: int = int(os.getenv("MOSANED_MAX_TURNS", "16"))

    # Refuse to serve if the emergency criteria carry no doctor's signature.
    # Off for local development; must be on for anything resembling a patient.
    require_reviewed_flags: bool = os.getenv("MOSANED_REQUIRE_REVIEW", "0") == "1"


settings = Settings()
