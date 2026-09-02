"""Domain objects. These are the shapes every layer agrees on."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Field-level visibility. The same history renders two ways from one record;
# this is also where consent and data-protection scoping will hang later.
PATIENT = "patient"
CLINICIAN = "clinician"


class CareLevel(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class MessageKind(str, Enum):
    """What a patient's message is asking for. The fork at the front."""
    SYMPTOM = "symptom"      # something they are experiencing -> take a history
    QUESTION = "question"    # something they want to understand -> answer it
    BOTH = "both"            # "my chest hurts, is that a heart attack?"


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True)
class KnowledgeAnswer:
    """An answer to a general question. `grounded` is false when nothing
    usable was found -- and then we say so rather than improvise."""
    text: str
    sources: list[Source]
    grounded: bool

    @property
    def usable(self) -> bool:
        return self.grounded and bool(self.text.strip())


class SessionState(str, Enum):
    GATHERING = "gathering"
    COMPLETE = "complete"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class FlagSpec:
    """One red flag the model is asked to sense as a yes/no."""
    id: str
    category: str
    level: str          # "emergency" | "urgent"
    sense: str


@dataclass(frozen=True)
class FiredFlag:
    flag_id: str
    category: str
    level: str
    on_message: int     # which turn it fired on
    quote: str          # what the patient actually said


@dataclass(frozen=True)
class FreeConcern:
    """The model's unconstrained second opinion. OR-ed with the flag list."""
    concerned: bool
    reason: str = ""


@dataclass(frozen=True)
class GateAssessment:
    """One read of a patient message: which named flags are present, plus the
    model's own unanchored judgment. Both travel in a single response, but they
    stay independent signals -- the OR that combines them lives in the engine."""
    present_flag_ids: list[str]
    free_concern: FreeConcern
    kind: "MessageKind" = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GateResult:
    escalate: bool
    fired: list[FiredFlag] = field(default_factory=list)
    free_concern: FreeConcern | None = None
    # True when the message could not be read for danger at all. Not the same
    # as "nothing found": we halt rather than let an unread message through.
    read_failed: bool = False
    # Symptom or question, read in the same pass. A separate call for this was
    # a third of the quota spent re-reading a message we had just read.
    kind: "MessageKind" = None  # type: ignore[assignment]

    @property
    def emergency_flags(self) -> list[FiredFlag]:
        return [f for f in self.fired if f.level == "emergency"]

    @property
    def urgent_flags(self) -> list[FiredFlag]:
        return [f for f in self.fired if f.level == "urgent"]

    @property
    def fired_categories(self) -> set[str]:
        return {f.category for f in self.fired}


@dataclass(frozen=True)
class Slot:
    id: str
    about: str
    critical: bool = False
    type: str = "text"
    ask_if: dict[str, Any] | None = None
    derive: dict[str, Any] | None = None
    is_background: bool = False


@dataclass
class Routing:
    specialty: str
    care_level: CareLevel
    reason: str
    forced_by_rule: bool = False


@dataclass
class StructuredHistory:
    """The object everything downstream reads."""
    session_id: str
    presenting_complaint_raw: str = ""
    presenting_complaint_category: str = "unknown"
    flow_id: str = "generic"
    hpi: dict[str, Any] = field(default_factory=dict)
    background: dict[str, Any] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)
    pertinent_negatives: list[str] = field(default_factory=list)
    red_flags_fired: list[FiredFlag] = field(default_factory=list)
    care_level: CareLevel = CareLevel.ROUTINE
    suggested_specialty: str = ""
    routing_reason: str = ""
    clinician_notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Which fields a patient may see. Anything absent is clinician-only.
    _PATIENT_VISIBLE = {
        "presenting_complaint_raw", "presenting_complaint_category", "hpi",
        "background", "derived", "pertinent_negatives", "red_flags_fired",
        "care_level", "suggested_specialty", "routing_reason", "created_at",
    }

    def to_patient_view(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if k in self._PATIENT_VISIBLE}

    def to_clinician_view(self) -> dict[str, Any]:
        return asdict(self)
