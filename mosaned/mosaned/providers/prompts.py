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
    "You read a patient conversation for danger signs, in this order.\n"
    "\n"
    "First: say whether this person must be seen WITHIN HOURS -- an emergency "
    "department or an ambulance, tonight. That is the only thing this question "
    "asks.\n"
    "\n"
    "Say yes for: failing airway or breathing, collapse, sudden neurological "
    "deficit, uncontrolled bleeding, anaphylaxis, signs of sepsis, an acute "
    "abdomen, intent to end their life.\n"
    "\n"
    "Say NO for anything serious that needs prompt investigation over the coming "
    "days rather than tonight. A months-long cough, blood-streaked phlegm, "
    "unexplained weight loss, a suspicious lump -- these deserve urgency and may "
    "well turn out to be grave, but the answer here is still no, because sending "
    "this person to an emergency department tonight helps them less than an "
    "urgent clinic appointment does. The named warning signs below already carry "
    "that urgency; if your concern is one of them, list it there and answer no "
    "here.\n"
    "\n"
    "A short, vague, or partial answer is NOT a reason to say yes. Missing "
    "information is NOT a reason to say yes. Most ordinary complaints are not "
    "emergencies.\n"
    "\n"
    "Then: list which of the NAMED warning signs below are present. Each of those "
    "is a specific question, so there you should err the other way -- if a named "
    "sign is plausibly present, list it.\n"
    "\n"
    "Finally: say what kind of message it is. 'symptom' if they are describing "
    "something they are experiencing. 'question' if they want to understand "
    "something general and are not reporting a problem of their own right now. "
    "'both' if they do each in the same message.\n"
    "\n"
    "Do not diagnose, advise, or explain."
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
            "kind": {"type": "string", "enum": ["symptom", "question", "both"]},
        },
        "required": ["concerned", "concern_reason", "present", "kind"],
        "additionalProperties": False,
    }


def assess_prompt(message: str, flags: list[FlagSpec], context: str = "") -> str:
    """A message is judged in its conversation, not alone. "It's dry" means
    nothing by itself; as an answer about a three-day cough it means a great
    deal, and judging it without that is asking an unanswerable question."""
    lines = "\n".join(f"- {f.id}: {f.sense}" for f in flags)
    conversation = f"Conversation so far:\n{context}\n\n" if context else ""
    # Order matters for speed, not just readability. The flag list is identical
    # on every call and is by far the largest part of the prompt, so it goes
    # FIRST: a runtime that caches a stable prefix can reuse it across the whole
    # session. The conversation and the new message vary, so they go last.
    # Reversed, every message pays full prompt-processing cost for all 39 flags.
    return (
        f"Named warning signs:\n{lines}\n\n"
        f"{conversation}"
        f"Latest patient message:\n\"\"\"{message}\"\"\""
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


def extract_prompt(message: str, slots: list[Slot], asked: str = "",
                   answering: str = "") -> str:
    """A reply is meaningless without its question. "no i dont" answers
    whichever field was just asked about; handed over on its own it answers
    nothing, and the patient gets asked again -- which is exactly what happened.
    """
    lines = "\n".join(f"- {s.id}: {s.about}" for s in slots)
    context = ""
    if asked:
        context = (
            f"You have just asked the patient:\n\"\"\"{asked}\"\"\"\n"
            f"Their reply below is most likely an answer to that"
        )
        if answering:
            context += f", which is the field `{answering}`"
        context += (
            ". Short replies such as \"no\", \"yes\", \"i don't know\" or "
            "\"sometimes\" are real answers to it -- record them as given. "
            "Only leave the field empty if the reply genuinely does not "
            "address the question at all.\n\n"
        )
    return (
        f"{context}"
        f"Patient reply:\n\"\"\"{message}\"\"\"\n\n"
        f"Fields you may fill (only those this reply answers):\n{lines}"
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


# ---------------------------------------------------------------------------
# The fork: is this something they are feeling, or something they want to know?
# ---------------------------------------------------------------------------

INTENT_SYSTEM = (
    "You sort a patient's message into one of three kinds.\n"
    "\n"
    "symptom  - they are describing something they are experiencing or worried "
    "about in themselves or someone they are with.\n"
    "question - they want to understand something general: a medicine, a "
    "condition, a test, what a word means. They are not reporting a problem "
    "of their own right now.\n"
    "both     - they describe something they are experiencing AND ask a "
    "general question about it in the same message.\n"
    "\n"
    "Answer with the kind only."
)

INTENT_SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": ["symptom", "question", "both"]}},
    "required": ["kind"],
    "additionalProperties": False,
}


def intent_prompt(message: str) -> str:
    return f'Patient message:\n"""{message}"""'


# ---------------------------------------------------------------------------
# The information path. Deliberately receives the question and NOTHING else --
# no history, no chart, no complaint. See engine/knowledge.py for why.
# ---------------------------------------------------------------------------

KNOWLEDGE_SYSTEM = (
    "You answer a general health question for a patient, in plain language, "
    "warmly and briefly.\n"
    "\n"
    "Search the trusted sources available to you and answer from what you find. "
    "Name the source. If you cannot find something that answers it, say you "
    "don't have reliable information on it rather than answering from memory.\n"
    "\n"
    "You are explaining a topic, never a person. You do not know anything about "
    "who is asking, and you must not guess: no 'in your case', no 'given your "
    "symptoms', no assessment of how serious their situation is. If the question "
    "invites you to judge their particular case, answer the general part and say "
    "plainly that only a doctor who examines them can answer the rest.\n"
    "\n"
    "Never diagnose. Never tell anyone to start, stop, or change a medicine -- "
    "point them to their doctor or pharmacist instead. Where a symptom would "
    "genuinely warrant being seen, say so without dramatising it.\n"
    "\n"
    "Two to four short paragraphs. No lists unless the answer is genuinely a list."
)


def knowledge_prompt(question: str) -> str:
    return f'Question:\n"""{question}"""'
