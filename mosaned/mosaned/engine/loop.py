"""Slot filling.

The engine decides which question comes next and when the history is done.
The model only words the question and reads the answer. Derived values and
clinician notes are computed by rule, never asked for and never generated.
"""
from __future__ import annotations

import re
from typing import Any

from ..domain import Slot, StructuredHistory

_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)", re.I
)
_PER_WEEK = {"day": 1 / 7, "week": 1.0, "month": 4.345, "year": 52.18}


def duration_in_weeks(text: str) -> float | None:
    """Read a stated duration as weeks. Returns None if the text has no number."""
    match = _DURATION.search(text or "")
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower().rstrip("s")
    return amount * _PER_WEEK[unit]


def applicable(slot: Slot, filled: dict[str, Any]) -> bool:
    """Conditional slots: only ask if the gating answer says so."""
    if not slot.ask_if:
        return True
    gate_value = str(filled.get(slot.ask_if["slot"], "")).lower()
    return any(cue.lower() in gate_value for cue in slot.ask_if.get("contains", []))


def next_slot(flow: dict, history: StructuredHistory) -> Slot | None:
    """The next unanswered critical slot, or None when the history is complete.

    Slots the patient has already said they cannot answer are skipped. Without
    that, "I don't know" leaves the conversation asking the same question until
    it hits the turn limit -- and people say "I don't know" constantly.
    """
    filled = {**history.hpi, **history.background}
    for slot in [*flow["slots"], *flow["background"]]:
        if not slot.critical:
            continue
        if slot.id in filled and str(filled[slot.id]).strip():
            continue
        if slot.id in history.not_known:
            continue
        if not applicable(slot, filled):
            continue
        return slot
    return None


def record(history: StructuredHistory, flow: dict, extracted: dict[str, Any]) -> None:
    """Write extracted values into the right half of the history."""
    background_ids = {s.id for s in flow["background"]}
    for key, value in extracted.items():
        if not str(value).strip():
            continue
        if key in background_ids:
            history.background[key] = value
        else:
            history.hpi[key] = value


def apply_derivations(history: StructuredHistory, flow: dict) -> None:
    """Rule-computed fields. Thresholds are compared in Python, never by the model."""
    for slot in [*flow["slots"], *flow["background"]]:
        if not slot.derive:
            continue
        source = history.hpi.get(slot.id) or history.background.get(slot.id)
        if not source:
            continue
        weeks = duration_in_weeks(str(source))
        if weeks is None:
            continue
        for rule in slot.derive.get("rules", []):
            if "under_weeks" in rule and weeks < rule["under_weeks"]:
                history.derived[slot.derive["field"]] = rule["value"]
                break
            if "else" in rule:
                history.derived[slot.derive["field"]] = rule["else"]
                break


def apply_clinician_notes(history: StructuredHistory, flow: dict) -> None:
    """Clinician-only considerations. Rule-selected text, never model-written,
    and never rendered to the patient."""
    values = {**history.hpi, **history.background, **history.derived}
    fired_ids = {f.flag_id for f in history.red_flags_fired}

    for rule in flow.get("clinician_note_rules", []):
        note = rule.get("note", "")
        if note in history.clinician_notes:
            continue

        if "when_slot_contains" in rule:
            spec = rule["when_slot_contains"]
            haystack = str(values.get(spec["slot"], "")).lower()
            if haystack and any(cue.lower() in haystack for cue in spec["any_of"]):
                history.clinician_notes.append(note)

        elif "when_slot_equals" in rule:
            spec = rule["when_slot_equals"]
            if values.get(spec["slot"]) == spec["value"]:
                history.clinician_notes.append(note)

        elif "when_flag_fired" in rule:
            if rule["when_flag_fired"] in fired_ids:
                history.clinician_notes.append(note)


def pertinent_negatives(history: StructuredHistory, flow: dict) -> list[str]:
    """Critical topics that were asked about and came back clear. This is the
    part that makes a history useful to a doctor rather than just a transcript."""
    negatives: list[str] = []
    values = {**history.hpi, **history.background}
    for slot in [*flow["slots"], *flow["background"]]:
        if not slot.critical:
            continue
        answer = str(values.get(slot.id, "")).strip().lower()
        if answer and _is_denial(answer):
            negatives.append(slot.id)
    return negatives


_DENIALS = {
    "no", "none", "nothing", "never", "not", "denies", "nope", "nil", "n/a", "na",
}


def _is_denial(answer: str) -> bool:
    """True when the answer opens with a denial. Compares the first word so
    that 'no', 'no,' and 'no.' all read the same."""
    first = re.split(r"[\s,.;:!]+", answer, maxsplit=1)[0]
    return first in _DENIALS
