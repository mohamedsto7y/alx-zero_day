"""Shared behaviour for real (network-backed) providers.

Subclasses implement `_json_call` only. Everything above re-validates whatever
comes back: a provider's output is data from outside the system, so the engine
never trusts its shape, only its content.
"""
from __future__ import annotations

import time
from typing import Any

from ..config import settings
from ..domain import (
    FlagSpec, FreeConcern, GateAssessment, KnowledgeAnswer, MessageKind, Slot,
)
from . import prompts


class JSONProviderBase:
    name = "base"

    def _json_call(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _timed(self, label: str, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        if not settings.debug_timing:
            return self._json_call(system, user, schema)
        start = time.perf_counter()
        try:
            return self._json_call(system, user, schema)
        finally:
            print(f"  [{label}: {time.perf_counter() - start:.1f}s]", flush=True)

    def assess(self, message: str, flags: list[FlagSpec], context: str = "") -> GateAssessment:
        raw = self._timed(
            "gate",
            prompts.ASSESS_SYSTEM,
            prompts.assess_prompt(message, flags, context),
            prompts.assess_schema(flags),
        )
        known = {f.id for f in flags}
        present = [fid for fid in raw.get("present", []) or [] if fid in known]
        return GateAssessment(
            present_flag_ids=present,
            free_concern=FreeConcern(
                concerned=bool(raw.get("concerned")),
                reason=str(raw.get("concern_reason", ""))[:400],
            ),
        )

    def classify_complaint(self, message: str, categories: list[str]) -> str:
        raw = self._timed(
            "classify",
            prompts.CLASSIFY_SYSTEM,
            prompts.classify_prompt(message, categories),
            prompts.classify_schema(categories),
        )
        got = str(raw.get("category", "unknown"))
        return got if got in categories else "unknown"

    def phrase_question(self, slot: Slot, complaint: str, already_asked: list[str]) -> str:
        raw = self._timed(
            "question",
            prompts.QUESTION_SYSTEM,
            prompts.question_prompt(slot, complaint, already_asked),
            prompts.QUESTION_SCHEMA,
        )
        question = str(raw.get("question", "")).strip()
        return question or f"Can you tell me about {slot.about}?"

    def extract(self, message: str, slots: list[Slot]) -> dict[str, Any]:
        raw = self._timed(
            "extract",
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
        raw = self._timed(
            "specialty",
            prompts.SPECIALTY_SYSTEM,
            prompts.specialty_prompt(summary, specialties),
            prompts.specialty_schema(specialties),
        )
        got = str(raw.get("specialty", ""))
        return got if got in specialties else ""

    def classify_intent(self, message: str) -> MessageKind:
        raw = self._timed(
            "intent", prompts.INTENT_SYSTEM, prompts.intent_prompt(message),
            prompts.INTENT_SCHEMA,
        )
        try:
            return MessageKind(str(raw.get("kind", "symptom")))
        except ValueError:
            # An unrecognised answer means we take a history, which is the
            # safer default: it keeps the emergency gate in the loop.
            return MessageKind.SYMPTOM

    def answer_question(self, question: str, domains: list[str]) -> KnowledgeAnswer:
        """Providers without a search tool do not answer from memory."""
        return KnowledgeAnswer(text="", sources=[], grounded=False)
