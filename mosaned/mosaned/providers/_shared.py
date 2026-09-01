"""Shared behaviour for real (network-backed) providers.

Subclasses implement `_json_call` only. Everything above re-validates whatever
comes back: a provider's output is data from outside the system, so the engine
never trusts its shape, only its content.
"""
from __future__ import annotations

from typing import Any

from ..domain import FlagSpec, FreeConcern, Slot
from . import prompts


class JSONProviderBase:
    name = "base"

    def _json_call(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def sense_flags(self, message: str, flags: list[FlagSpec]) -> dict[str, bool]:
        raw = self._json_call(
            prompts.SENSE_SYSTEM,
            prompts.sense_prompt(message, flags),
            prompts.sense_schema(flags),
        )
        # Anything missing or non-boolean is treated as not sensed; the free
        # concern pass and the urgent-flag review are the backstop.
        return {f.id: bool(raw.get(f.id)) for f in flags}

    def free_concern(self, message: str) -> FreeConcern:
        raw = self._json_call(
            prompts.CONCERN_SYSTEM, prompts.concern_prompt(message), prompts.CONCERN_SCHEMA
        )
        return FreeConcern(
            concerned=bool(raw.get("concerned")),
            reason=str(raw.get("reason", ""))[:400],
        )

    def classify_complaint(self, message: str, categories: list[str]) -> str:
        raw = self._json_call(
            prompts.CLASSIFY_SYSTEM,
            prompts.classify_prompt(message, categories),
            prompts.classify_schema(categories),
        )
        got = str(raw.get("category", "unknown"))
        return got if got in categories else "unknown"

    def phrase_question(self, slot: Slot, complaint: str, already_asked: list[str]) -> str:
        raw = self._json_call(
            prompts.QUESTION_SYSTEM,
            prompts.question_prompt(slot, complaint, already_asked),
            prompts.QUESTION_SCHEMA,
        )
        question = str(raw.get("question", "")).strip()
        return question or f"Can you tell me about {slot.about}?"

    def extract(self, message: str, slots: list[Slot]) -> dict[str, Any]:
        raw = self._json_call(
            prompts.EXTRACT_SYSTEM,
            prompts.extract_prompt(message, slots),
            prompts.extract_schema(slots),
        )
        allowed = {s.id for s in slots}
        return {
            k: str(v).strip()
            for k, v in raw.items()
            if k in allowed and isinstance(v, (str, int, float)) and str(v).strip()
        }

    def propose_specialty(self, summary: str, specialties: list[str]) -> str:
        raw = self._json_call(
            prompts.SPECIALTY_SYSTEM,
            prompts.specialty_prompt(summary, specialties),
            prompts.specialty_schema(specialties),
        )
        got = str(raw.get("specialty", ""))
        return got if got in specialties else ""
