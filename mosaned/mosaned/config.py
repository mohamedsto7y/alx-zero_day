"""Settings. Everything the prototype needs to change lives here or in the env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: lets settings live in a .env file instead of the shell
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass

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
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    # Haiku 4.5 is the cheap, fast tier -- roughly a cent or two per intake.
    # Change this one value for a stronger model.
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

    # The only curation the information path needs: a list of domains it may
    # search. No corpus, no scraping, nothing to keep up to date.
    source_domains: str = os.getenv(
        "MOSANED_SOURCE_DOMAINS",
        "nhs.uk,msdmanuals.com,who.int,mayoclinic.org,cdc.gov",
    )
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

    lang: str = os.getenv("MOSANED_LANG", "en")
    db_path: str = os.getenv("MOSANED_DB", str(Path.cwd() / "mosaned.db"))

    # Hard stop so a stuck conversation can't loop forever.
    max_turns: int = int(os.getenv("MOSANED_MAX_TURNS", "16"))

    # Let the model word each question. Warmer, but it costs a call per turn --
    # noticeable on a local model, free on a hosted one.
    phrase_questions: bool = os.getenv("MOSANED_PHRASE_QUESTIONS", "0") == "1"

    # Print how long each model call took.
    debug_timing: bool = os.getenv("MOSANED_DEBUG_TIMING", "0") == "1"

    # Refuse to serve if the emergency criteria carry no doctor's signature.
    # Off for local development; must be on for anything resembling a patient.
    require_reviewed_flags: bool = os.getenv("MOSANED_REQUIRE_REVIEW", "0") == "1"


settings = Settings()
