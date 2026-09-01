"""The one surface the engine sees.

Every method is a narrow, constrained language task. None of them decides
anything: the model senses, classifies, phrases and extracts. Decisions are
made in the engine, in Python. Swapping providers must never change what the
system decides -- only how well it reads what a patient wrote.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..domain import FlagSpec, FreeConcern, Slot


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def sense_flags(self, message: str, flags: list[FlagSpec]) -> dict[str, bool]:
        """Fill a fixed boolean per red flag. Sensor only, never a decision."""

    def free_concern(self, message: str) -> FreeConcern:
        """Unconstrained second opinion, OR-ed with the flag list."""

    def classify_complaint(self, message: str, categories: list[str]) -> str:
        """Pick one category from the list, or 'unknown'. Cannot invent one."""

    def phrase_question(self, slot: Slot, complaint: str, already_asked: list[str]) -> str:
        """Word the next question naturally. Free text, but only ever a question."""

    def extract(self, message: str, slots: list[Slot]) -> dict[str, Any]:
        """Pull every field this message answers into the named schema."""

    def propose_specialty(self, summary: str, specialties: list[str]) -> str:
        """Propose one specialty from the list. The engine may overrule it."""
