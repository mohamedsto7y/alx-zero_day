"""Prompts and the JSON schemas that constrain every reply.

Kept apart from transport so Ollama and Gemini share one wording, and so the
whole prompt surface can be translated in one place when the product moves to
Arabic. Every task here is narrow on purpose: the model is asked to read
language, never to decide anything.
"""
from __future__ import annotations

from typing import Any

from ..domain import FlagSpec, Slot

SENSE_SYSTEM = (
    "You read a patient's message and report whether specific warning signs are present. "
    "Answer only true or false for each. Do not diagnose, advise, or explain. "
    "If a sign is plausibly present but you are unsure, answer true: a false alarm is "
    "harmless here and a missed sign is not."
)

CONCERN_SYSTEM = (
    "You read a patient's message and say whether anything in it would worry an "
    "experienced emergency clinician enough to send this person for immediate care. "
    "Use everything you know; you are not limited to any list. Do not diagnose. "
    "If unsure, say you are concerned."
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


def sense_schema(flags: list[FlagSpec]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {f.id: {"type": "boolean"} for f in flags},
        "required": [f.id for f in flags],
        "additionalProperties": False,
    }


def sense_prompt(message: str, flags: list[FlagSpec]) -> str:
    lines = "\n".join(f"- {f.id}: {f.sense}" for f in flags)
    return f"Patient message:\n\"\"\"{message}\"\"\"\n\nWarning signs to report on:\n{lines}"


CONCERN_SCHEMA = {
    "type": "object",
    "properties": {
        "concerned": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["concerned", "reason"],
    "additionalProperties": False,
}


def concern_prompt(message: str) -> str:
    return f"Patient message:\n\"\"\"{message}\"\"\"\n\nWould this need immediate care?"


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
