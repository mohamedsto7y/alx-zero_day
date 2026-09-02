"""A deterministic test double.

Not a model. It exists so the engine's decisions can be tested without a
network, and so the loop can be demonstrated on a machine with nothing
installed. Its extraction is deliberately crude -- if a test passes here it is
testing the engine's logic, which is the point.
"""
from __future__ import annotations

from typing import Any

from ..domain import (
    FlagSpec, FreeConcern, GateAssessment, KnowledgeAnswer, MessageKind, Slot,
)

# Crude cue words per flag, for offline testing only.
_CUES: dict[str, tuple[str, ...]] = {
    "chest_pain_severe": ("crushing chest", "chest pain", "pressure in my chest", "chest is crushing"),
    "breathless_at_rest": ("can't breathe", "cannot breathe", "breathless at rest", "struggling to breathe"),
    "cannot_speak_full_sentences": ("can't finish a sentence", "cannot speak", "can't get my words out"),
    "cyanosis": ("blue lips", "lips are blue", "turning blue"),
    "choking_or_inhaled_object": ("choking", "swallowed something", "inhaled"),
    "haemoptysis_significant": ("coughing up a lot of blood", "mouthful of blood", "cup of blood"),
    "haemoptysis_minor": ("bit of blood", "streaks of blood", "specks of blood",
                          "little blood", "blood in it", "blood in the phlegm",
                          "coughing up blood", "blood when i cough"),
    "stroke_signs": ("face is drooping", "one side", "can't move my arm", "slurred"),
    "thunderclap_headache": ("worst headache", "sudden severe headache"),
    "seizure": ("seizure", "fit", "convulsion"),
    "altered_consciousness": ("confused", "won't wake", "unresponsive"),
    "airway_swelling": ("throat is closing", "tongue is swollen", "lips swollen"),
    "suicidal_intent": ("end my life", "kill myself", "suicidal"),
    "uncontrolled_bleeding": ("bleeding heavily", "won't stop bleeding"),
    "haematemesis_or_melaena": ("vomiting blood", "black stool", "tarry"),
    "persistent_cough": ("weeks", "month", "months"),
    "unintended_weight_loss": ("losing weight", "lost weight"),
    "night_sweats": ("night sweats", "sweating at night"),
    "progressive_breathlessness": ("getting more breathless", "worse breathing"),
}

_COMPLAINT_CUES: dict[str, tuple[str, ...]] = {
    "cough": ("cough", "coughing", "phlegm", "sputum"),
    "chest_pain": ("chest pain", "chest hurts"),
    "headache": ("headache", "head hurts", "migraine"),
    "abdominal_pain": ("stomach", "belly", "abdomen", "tummy"),
    "fever": ("fever", "temperature", "shivery"),
    "rash": ("rash", "spots", "itchy skin"),
    "back_pain": ("back pain", "back hurts"),
    "dizziness": ("dizzy", "lightheaded", "vertigo"),
}


class StubProvider:
    name = "stub"

    def assess(self, message: str, flags: list[FlagSpec], context: str = "") -> GateAssessment:
        low = message.lower()
        present = [f.id for f in flags if any(cue in low for cue in _CUES.get(f.id, ()))]
        # The stub has no clinical judgment, so it never raises a concern of its
        # own: tests exercise the flag list rather than that branch.
        return GateAssessment(
            present_flag_ids=present,
            free_concern=FreeConcern(concerned=False),
            kind=self.classify_intent(message),
        )

    def classify_complaint(self, message: str, categories: list[str]) -> str:
        low = message.lower()
        for cat, cues in _COMPLAINT_CUES.items():
            if cat in categories and any(cue in low for cue in cues):
                return cat
        return "unknown"

    def phrase_question(self, slot: Slot, complaint: str, already_asked: list[str]) -> str:
        return f"Can you tell me about {slot.about}?"

    def extract(self, message: str, slots: list[Slot], asked: str = "",
                answering: str = "") -> dict[str, Any]:
        # Assigns the whole message to the first slot asked for. Enough to drive
        # the loop deterministically; nothing like real extraction.
        return {slots[0].id: message.strip()} if slots and message.strip() else {}

    def propose_specialty(self, summary: str, specialties: list[str]) -> str:
        low = summary.lower()
        for cue, spec in (
            ("cough", "pulmonology"), ("breath", "pulmonology"),
            ("chest", "cardiology"), ("head", "neurology"),
            ("stomach", "gastroenterology"), ("skin", "dermatology"),
            ("rash", "dermatology"), ("back", "orthopedics"),
        ):
            if cue in low and spec in specialties:
                return spec
        return specialties[0] if specialties else ""

    def classify_intent(self, message: str) -> MessageKind:
        low = message.lower()
        asking = any(cue in low for cue in (
            "is it", "what is", "what are", "what does", "can i", "should i",
            "why do", "how do", "does it", "safe to", "?",
        ))
        describing = any(cue in low for cue in (
            "i have", "i've had", "i feel", "my ", "i am", "i'm",
        ))
        if asking and describing:
            return MessageKind.BOTH
        if asking:
            return MessageKind.QUESTION
        return MessageKind.SYMPTOM

    def answer_question(self, question: str, domains: list[str]) -> KnowledgeAnswer:
        # The stub cannot search, so it declines rather than inventing.
        return KnowledgeAnswer(text="", sources=[], grounded=False)
