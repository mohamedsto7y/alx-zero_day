"""The one surface the engine sees.

Every method is a narrow, constrained language task. None of them decides
anything: the model senses, classifies, phrases and extracts. Decisions are
made in the engine, in Python. Swapping providers must never change what the
system decides -- only how well it reads what a patient wrote.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..domain import FlagSpec, GateAssessment, Slot


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def assess(self, message: str, flags: list[FlagSpec]) -> GateAssessment:
        """Read one message for danger: which named flags are present, and the
        model's own unanchored judgment. Sensor only -- it never decides
        whether to escalate."""

    def classify_complaint(self, message: str, categories: list[str]) -> str:
        """Pick one category from the list, or 'unknown'. Cannot invent one."""

    def phrase_question(self, slot: Slot, complaint: str, already_asked: list[str]) -> str:
        """Word the next question naturally. Free text, but only ever a question."""

    def extract(self, message: str, slots: list[Slot]) -> dict[str, Any]:
        """Pull every field this message answers into the named schema."""

    def propose_specialty(self, summary: str, specialties: list[str]) -> str:
        """Propose one specialty from the list. The engine may overrule it."""
