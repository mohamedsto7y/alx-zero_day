"""The question loop, the flow tiers, and what comes out."""
from mosaned.domain import SessionState, StructuredHistory
from mosaned.engine.clinical import load_flow
from mosaned.engine.loop import (
    apply_clinician_notes, apply_derivations, next_slot, record,
)
from mosaned.engine.session import IntakeSession
from mosaned.providers.stub import StubProvider


class Quiet(StubProvider):
    """The stub already volunteers no concern of its own."""


def test_authored_flow_used_when_it_exists():
    session = IntakeSession(_provider=Quiet())
    session.send("I have a bad cough")
    assert session.flow["id"] == "cough"
    assert session.history.presenting_complaint_category == "cough"


def test_generic_frame_catches_everything_else():
    session = IntakeSession(_provider=Quiet())
    session.send("I've got an itchy rash on my arm")
    assert session.flow["id"] == "generic"
    assert session.state is SessionState.GATHERING


def test_unknown_complaint_still_gets_a_history():
    session = IntakeSession(_provider=Quiet())
    result = session.send("something just feels off lately")
    assert session.history.presenting_complaint_category == "unknown"
    assert session.flow["id"] == "generic"
    assert result["state"] == "gathering"


def test_conditional_slot_is_skipped_when_not_applicable():
    flow = load_flow("cough")
    history = StructuredHistory(session_id="t")
    history.hpi.update({"duration": "3 days", "dry_or_productive": "completely dry"})
    ids = []
    while (slot := next_slot(flow, history)) and len(ids) < 12:
        ids.append(slot.id)
        history.hpi[slot.id] = "no"
    assert "sputum" not in ids


def test_conditional_slot_is_asked_when_applicable():
    flow = load_flow("cough")
    history = StructuredHistory(session_id="t")
    history.hpi.update({"duration": "3 days", "dry_or_productive": "bringing up phlegm"})
    assert next_slot(flow, history).id == "sputum"


def test_duration_is_derived_by_rule_not_asked():
    flow = load_flow("cough")
    history = StructuredHistory(session_id="t")
    record(history, flow, {"duration": "about 10 weeks now"})
    apply_derivations(history, flow)
    assert history.derived["duration_category"] == "chronic"


def test_ambiguous_duration_derives_nothing_rather_than_guessing():
    flow = load_flow("cough")
    history = StructuredHistory(session_id="t")
    record(history, flow, {"duration": "ages, I can't remember"})
    apply_derivations(history, flow)
    assert "duration_category" not in history.derived


def test_ace_inhibitor_note_is_clinician_only():
    flow = load_flow("cough")
    history = StructuredHistory(session_id="t")
    record(history, flow, {"medications": "ramipril for blood pressure"})
    apply_clinician_notes(history, flow)
    assert any("ACE inhibitor" in n for n in history.clinician_notes)
    assert "clinician_notes" not in history.to_patient_view()
    assert "clinician_notes" in history.to_clinician_view()


def test_intake_terminates():
    session = IntakeSession(_provider=Quiet())
    session.send("I have a cough")
    for _ in range(30):
        if session.state is not SessionState.GATHERING:
            break
        session.send("no")
    assert session.state is SessionState.COMPLETE
    assert session.history.suggested_specialty


def test_pertinent_negatives_are_recorded():
    session = IntakeSession(_provider=Quiet())
    session.send("I have a cough")
    for _ in range(30):
        if session.state is not SessionState.GATHERING:
            break
        session.send("no, none of that")
    assert session.history.pertinent_negatives


def test_inapplicable_slot_is_never_offered_for_extraction():
    """A patient who says their cough is dry must not have sputum colour
    presented as a fillable field."""
    session = IntakeSession(_provider=Quiet())
    session.send("I have a cough")
    session.send("it is completely dry")
    filled = {**session.history.hpi, **session.history.background}
    assert "sputum" not in filled
