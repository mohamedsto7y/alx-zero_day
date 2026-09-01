"""Prompts and the JSON schemas that constrain every reply.

Kept apart from transport so Ollama and Gemini share one wording, and so the
whole prompt surface can be translated in one place when the product moves to
Arabic. Every task here is narrow on purpose: the model is asked to read
language, never to decide anything.
"""
from __future__ import annotations

from typing import Any

from ..domain import FlagSpec, Slot

ASSESS_SYSTEM = (
    "You read a patient's message for danger signs, in this order.\n"
    "First, using everything you know as an experienced emergency clinician and "
    "WITHOUT looking at any list, say whether anything here would make you send "
    "this person for immediate care.\n"
    "Then, separately, list which of the named warning signs are present. List only "
    "the ones actually present -- usually that is none.\n"
    "Do not diagnose, advise, or explain. When unsure either way, err toward "
    "concern: a false alarm is harmless here and a missed sign is not."
)

CLASSIFY_SYSTEM = (
    "You sort a patient's message into exactly one of the categories given. "
    "If none fits, answer 'unknown'. Never invent a category."
)

QUESTION_SYSTEM = (
    "You are taking a clinical history. Ask ONE short, warm, plain-language question "
    "about the topic given. Do not diagnose, reassure, advise, or mention possible "
    "causes. Ask the question and nothing else."
)

EXTRACT_SYSTEM = (
    "You pull facts out of a patient's message into named fields. Only fill a field "
    "the message actually answers; leave the rest out. Never guess, never infer a "
    "diagnosis, and copy the patient's own meaning rather than interpreting it."
)

SPECIALTY_SYSTEM = (
    "You suggest which medical specialty is the best fit for a patient's history. "
    "Choose exactly one from the list given. Never invent a specialty."
)


def assess_schema(flags: list[FlagSpec]) -> dict[str, Any]:
    """`concerned` is declared first so the model commits to its own judgment
    before it scans the list -- a scan that found nothing must not anchor it
    into saying it is unworried. `present` is a short array rather than a
    boolean per flag: on a local model that is the difference between emitting
    a handful of tokens and emitting one key/value pair for every flag we know."""
    return {
        "type": "object",
        "properties": {
            "concerned": {"type": "boolean"},
            "concern_reason": {"type": "string"},
            "present": {
                "type": "array",
                "items": {"type": "string", "enum": [f.id for f in flags]},
            },
        },
        "required": ["concerned", "concern_reason", "present"],
        "additionalProperties": False,
    }


def assess_prompt(message: str, flags: list[FlagSpec]) -> str:
    lines = "\n".join(f"- {f.id}: {f.sense}" for f in flags)
    return (
        f"Patient message:\n\"\"\"{message}\"\"\"\n\n"
        f"Named warning signs:\n{lines}"
    )


def classify_schema(categories: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"category": {"type": "string", "enum": [*categories, "unknown"]}},
        "required": ["category"],
        "additionalProperties": False,
    }


def classify_prompt(message: str, categories: list[str]) -> str:
    return (
        f"Patient message:\n\"\"\"{message}\"\"\"\n\n"
        f"Categories: {', '.join(categories)}"
    )


QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}


def question_prompt(slot: Slot, complaint: str, already_asked: list[str]) -> str:
    asked = "\n".join(f"- {q}" for q in already_asked[-4:]) or "- (nothing yet)"
    return (
        f"The patient came in about: {complaint}\n"
        f"Already asked:\n{asked}\n\n"
        f"Ask about: {slot.about}"
    )


def extract_schema(slots: list[Slot]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {s.id: {"type": "string"} for s in slots},
        "required": [],
        "additionalProperties": False,
    }


def extract_prompt(message: str, slots: list[Slot]) -> str:
    lines = "\n".join(f"- {s.id}: {s.about}" for s in slots)
    return (
        f"Patient message:\n\"\"\"{message}\"\"\"\n\n"
        f"Fields you may fill (only those this message answers):\n{lines}"
    )


def specialty_schema(specialties: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"specialty": {"type": "string", "enum": specialties}},
        "required": ["specialty"],
        "additionalProperties": False,
    }


def specialty_prompt(summary: str, specialties: list[str]) -> str:
    return f"History:\n{summary}\n\nSpecialties: {', '.join(specialties)}"
