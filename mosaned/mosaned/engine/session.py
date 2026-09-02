"""The orchestrator: one intake conversation, start to finish."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..domain import (
    CareLevel, GateResult, MessageKind, Routing, SessionState, StructuredHistory,
)
from ..providers import get_provider
from ..providers.base import LLMProvider
from ..strings import t
from . import gate as gate_mod
from . import knowledge as knowledge_mod
from . import loop as loop_mod
from . import routing as routing_mod
from .clinical import flags_reviewed_by, known_complaints, load_flow


class UnreviewedClinicalData(RuntimeError):
    """Raised when the emergency criteria carry no doctor's signature."""


def _plain_question(slot) -> str:
    """The flow's own wording, used when we aren't spending a model call on
    phrasing.

    A slot's `about` describes the topic to the model and is written in the
    third person -- read straight to a patient it produces "whether they have a
    fever", which is nobody's idea of a companion. `ask` is the patient-facing
    wording; the template is only a fallback for slots that lack one.
    """
    return slot.ask or t("ask.template", topic=slot.about)


@dataclass
class Turn:
    role: str          # "patient" | "system"
    text: str


@dataclass
class IntakeSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: SessionState = SessionState.GATHERING
    history: StructuredHistory = field(default=None)  # type: ignore[assignment]
    transcript: list[Turn] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    gate: GateResult | None = None
    routing: Routing | None = None
    flow: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    complaint_established: bool = False
    pending_slot_id: str | None = None
    slot_attempts: dict[str, int] = field(default_factory=dict)
    _provider: LLMProvider = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if settings.require_reviewed_flags and not flags_reviewed_by():
            raise UnreviewedClinicalData(t("error.unreviewed_flags"))
        if self._provider is None:
            self._provider = get_provider()
        if self.history is None:
            self.history = StructuredHistory(session_id=self.session_id)
        if not self.flow:
            self.flow = load_flow("generic")

    # ---- the conversation ------------------------------------------------

    def open(self) -> str:
        return f"{t('greeting')}\n\n{t('disclaimer')}"

    def send(self, message: str) -> dict[str, Any]:
        """One patient message in, one system reply out."""
        if self.state is SessionState.ESCALATED:
            # Anything said after the gate fires gets the same instruction.
            # No reassurance, no negotiation -- just the direction again.
            self.transcript.append(Turn("patient", message))
            reply = t("escalate.emergency.repeat")
            self.transcript.append(Turn("system", reply))
            return {"state": self.state.value, "reply": reply,
                    "care_level": self.history.care_level.value}
        if self.state is not SessionState.GATHERING:
            return self._render()

        self.turn_count += 1
        self.transcript.append(Turn("patient", message))

        # 1. The gate, before anything else and on every single message.
        latest = gate_mod.run_gate(
            message, self._provider, self.turn_count, self._gate_context()
        )
        self.gate = gate_mod.merge(self.gate, latest)
        self.history.red_flags_fired = list(self.gate.fired)

        if self.gate.escalate:
            return self._escalate()

        if latest.read_failed:
            # We could not check this message for danger, so we refuse to act
            # on it -- no extraction, no next question. But the session stays
            # open: a transient outage should cost one message, not the whole
            # conversation, and nothing unsafe got through.
            self.turn_count -= 1
            reply = (
                f"{t('escalate.unavailable.title')}\n\n"
                f"{t('escalate.unavailable.body')}"
            )
            self.transcript.append(Turn("system", reply))
            return {"state": self.state.value, "reply": reply,
                    "read_failed": True, "awaiting": self.pending_slot_id}

        # 2. The fork. Read in the same pass as the gate: a separate call for
        # it was a third of the quota spent re-reading the same message.
        if latest.kind is MessageKind.QUESTION:
            return self._answer_question(message)

        # 3. The first symptom message sets the complaint and picks the flow.
        if not self.complaint_established:
            self._establish_complaint(message)

        # 4. Read whatever this message answered, across every open slot.
        # Conditional slots that don't apply are withheld, so the model is
        # never offered a field the flow says shouldn't be asked -- e.g. sputum
        # colour for someone who has said their cough is dry.
        filled = {**self.history.hpi, **self.history.background}
        open_slots = [
            s for s in [*self.flow["slots"], *self.flow["background"]]
            if s.id not in filled and loop_mod.applicable(s, filled)
        ]
        if open_slots:
            try:
                extracted = self._provider.extract(
                    message, open_slots,
                    asked=self.asked[-1] if self.asked else "",
                    answering=self.pending_slot_id or "",
                )
            except Exception:
                extracted = {}
            loop_mod.record(self.history, self.flow, extracted)

        loop_mod.apply_derivations(self.history, self.flow)
        loop_mod.apply_clinician_notes(self.history, self.flow)

        # 4b. If we asked something and still don't have it, that is an answer
        # too. Ask once more, then record that they could not say and move on:
        # a patient who does not know cannot be made to know by repetition.
        moved_on = self._retire_unanswered_slot()

        # 5. Next question, or finish.
        if self.turn_count >= settings.max_turns:
            return self._complete()

        slot = loop_mod.next_slot(self.flow, self.history)
        if slot is None:
            return self._complete()

        question = _plain_question(slot)
        if settings.phrase_questions:
            try:
                question = self._provider.phrase_question(
                    slot, self.history.presenting_complaint_category, self.asked
                )
            except Exception:
                pass

        if moved_on:
            question = f"{t('ask.moving_on')} {question}"

        self.asked.append(question)
        self.slot_attempts[slot.id] = self.slot_attempts.get(slot.id, 0) + 1
        self.pending_slot_id = slot.id
        self.transcript.append(Turn("system", question))
        return {"state": self.state.value, "reply": question, "awaiting": slot.id}

    # ---- internals -------------------------------------------------------

    def _retire_unanswered_slot(self) -> bool:
        """Stop asking a question the patient cannot answer.

        Returns True when a slot was retired, so the next question can
        acknowledge it rather than ploughing on as if nothing happened.
        """
        pending = self.pending_slot_id
        if not pending:
            return False
        filled = {**self.history.hpi, **self.history.background}
        if str(filled.get(pending, "")).strip():
            return False
        if self.slot_attempts.get(pending, 0) < 2:
            return False

        # Belt and braces. If extraction missed it but the patient plainly said
        # no, record the denial rather than filing them as unsure. Being told
        # "I've noted you're not sure" after saying no twice is the kind of
        # thing that ends a conversation, and it happened.
        last = next(
            (turn.text for turn in reversed(self.transcript) if turn.role == "patient"),
            "",
        )
        if loop_mod.is_denial(last):
            loop_mod.record(self.history, self.flow, {pending: last.strip()})
            return False

        if pending not in self.history.not_known:
            self.history.not_known.append(pending)
        return True

    def _gate_context(self, turns: int = 6) -> str:
        """What the gate needs to judge the newest message. Without it the gate
        is asked whether a bare "it's dry" is an emergency, which is not a
        question anyone can answer."""
        if not self.transcript:
            return ""
        lines = [
            f"{'patient' if turn.role == 'patient' else 'assistant'}: {turn.text}"
            for turn in self.transcript[-turns - 1:-1]
        ]
        complaint = self.history.presenting_complaint_raw
        if complaint:
            lines.insert(0, f"(came in about: {complaint})")
        return "\n".join(lines)

    def _answer_question(self, message: str) -> dict[str, Any]:
        """Answer a general question, then carry on where we left off.

        THE FIREWALL: only `message` crosses into the knowledge path. Not
        self.history, not the complaint, not the flags. If this path could see
        the chart, a helpful model would tailor the answer to this patient --
        and a tailored answer is a diagnosis, arrived at by nobody's decision.
        """
        result = knowledge_mod.answer(message, self._provider)
        reply = knowledge_mod.render(result)

        if result.grounded and self.complaint_established:
            reply += "\n\n" + t("knowledge.not_about_you")

        # Put the patient back where they were, so a question doesn't derail
        # the history they were part-way through giving.
        if self.complaint_established and self.asked:
            reply += "\n\n" + self.asked[-1]

        self.transcript.append(Turn("system", reply))
        return {
            "state": self.state.value,
            "reply": reply,
            "kind": "question",
            "grounded": result.grounded,
            "sources": [s.url for s in result.sources],
            "awaiting": self.pending_slot_id,
        }

    def _establish_complaint(self, message: str) -> None:
        self.history.presenting_complaint_raw = message.strip()
        try:
            category = self._provider.classify_complaint(message, known_complaints())
        except Exception:
            category = "unknown"
        self.history.presenting_complaint_category = category
        self.flow = load_flow(category if category != "unknown" else "generic")
        self.history.flow_id = self.flow["id"]
        self.complaint_established = True

    def _escalate(self) -> dict[str, Any]:
        self.state = SessionState.ESCALATED
        self.history.care_level = CareLevel.EMERGENCY
        body = (
            f"{t('escalate.emergency.title')}\n\n"
            f"{t('escalate.emergency.body')}\n\n"
            f"{t('escalate.emergency.footer')}"
        )
        self.transcript.append(Turn("system", body))
        return {
            "state": self.state.value,
            "reply": body,
            "care_level": CareLevel.EMERGENCY.value,
            "flags": [f.flag_id for f in self.gate.fired] if self.gate else [],
        }

    def _complete(self) -> dict[str, Any]:
        self.state = SessionState.COMPLETE
        self.history.pertinent_negatives = loop_mod.pertinent_negatives(self.history, self.flow)
        self.routing = routing_mod.decide(self.history, self.gate, self.flow, self._provider)
        self.history.care_level = self.routing.care_level
        self.history.suggested_specialty = self.routing.specialty
        self.history.routing_reason = self.routing.reason

        reply = f"{t('complete.title')}\n\n{t('complete.body')}"
        if self.routing.care_level is CareLevel.URGENT:
            reply += f"\n\n{t('complete.urgent_note')}"
        reply += "\n\n" + t("routing.suggested", specialty=self.routing.specialty.replace("_", " "))

        self.transcript.append(Turn("system", reply))
        return self._render(reply)

    def _render(self, reply: str | None = None) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reply": reply or "",
            "care_level": self.history.care_level.value,
            "specialty": self.history.suggested_specialty,
            "history": self.history.to_patient_view(),
        }
