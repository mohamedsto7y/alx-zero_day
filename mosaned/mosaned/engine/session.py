"""The orchestrator: one intake conversation, start to finish."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..domain import CareLevel, GateResult, Routing, SessionState, StructuredHistory
from ..providers import get_provider
from ..providers.base import LLMProvider
from ..strings import t
from . import gate as gate_mod
from . import loop as loop_mod
from . import routing as routing_mod
from .clinical import flags_reviewed_by, known_complaints, load_flow


class UnreviewedClinicalData(RuntimeError):
    """Raised when the emergency criteria carry no doctor's signature."""


def _plain_question(slot) -> str:
    """The flow's own wording, used when we aren't spending a model call on
    phrasing. A flow may carry an explicit `ask`; otherwise the slot's topic
    reads well enough on its own."""
    return t("ask.template", topic=slot.about)


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

        # 2. First message also sets the complaint and picks the flow.
        if self.turn_count == 1:
            self._establish_complaint(message)

        # 3. Read whatever this message answered, across every open slot.
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
                extracted = self._provider.extract(message, open_slots)
            except Exception:
                extracted = {}
            loop_mod.record(self.history, self.flow, extracted)

        loop_mod.apply_derivations(self.history, self.flow)
        loop_mod.apply_clinician_notes(self.history, self.flow)

        # 4. Next question, or finish.
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

        self.asked.append(question)
        self.transcript.append(Turn("system", question))
        return {"state": self.state.value, "reply": question, "awaiting": slot.id}

    # ---- internals -------------------------------------------------------

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

    def _establish_complaint(self, message: str) -> None:
        self.history.presenting_complaint_raw = message.strip()
        try:
            category = self._provider.classify_complaint(message, known_complaints())
        except Exception:
            category = "unknown"
        self.history.presenting_complaint_category = category
        self.flow = load_flow(category if category != "unknown" else "generic")
        self.history.flow_id = self.flow["id"]

    def _escalate(self) -> dict[str, Any]:
        self.state = SessionState.ESCALATED
        self.history.care_level = CareLevel.EMERGENCY
        if self.gate and self.gate.read_failed and not self.gate.fired:
            # We halted because we could not check, not because we found
            # something. Saying otherwise would be a lie to a worried person.
            body = (
                f"{t('escalate.unavailable.title')}\n\n"
                f"{t('escalate.unavailable.body')}"
            )
        else:
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
