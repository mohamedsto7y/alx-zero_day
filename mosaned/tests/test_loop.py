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


class CannotExtract(Quiet):
    """Matches what a real model does with "i dont know": it fills nothing.
    The stub's default would record the phrase itself as the answer, which
    hides the very loop this is about."""
    def extract(self, message, slots):
        return {}


def test_i_dont_know_does_not_loop_forever():
    """People say "I don't know" constantly. Asking again until the turn limit
    is not a conversation."""
    session = IntakeSession(_provider=CannotExtract())
    session.send("I have a cough")
    asked = []
    for _ in range(8):
        if session.state is not SessionState.GATHERING:
            break
        result = session.send("i dont know")
        asked.append(result.get("awaiting"))

    # No slot is asked more than twice before we accept they cannot say.
    for slot_id in set(asked):
        assert asked.count(slot_id) <= 2, f"{slot_id} was asked {asked.count(slot_id)} times"
    assert session.history.not_known, "unanswerable slots should be recorded"


def test_not_knowing_is_recorded_for_the_doctor():
    """"Patient could not say" is real clinical information, not a gap."""
    session = IntakeSession(_provider=CannotExtract())
    session.send("I have a cough")
    for _ in range(6):
        if session.state is not SessionState.GATHERING:
            break
        session.send("i dont know")

    view = session.history.to_patient_view()
    assert "not_known" in view
    assert session.history.not_known


def test_moving_on_is_acknowledged():
    session = IntakeSession(_provider=CannotExtract())
    session.send("I have a cough")
    replies = []
    for _ in range(5):
        if session.state is not SessionState.GATHERING:
            break
        replies.append(session.send("i dont know")["reply"])
    assert any("noted you're not sure" in r for r in replies)


def test_a_denial_is_never_filed_as_uncertainty():
    """"I told you I don't have it, I didn't say I'm not sure" -- a real
    patient, rightly furious. Even when extraction misses, a plain "no" is
    recorded as a denial, never as "not sure"."""
    session = IntakeSession(_provider=CannotExtract())
    session.send("I have a cough")
    replies = [session.send("no")["reply"] for _ in range(4)
               if session.state is SessionState.GATHERING]

    assert not any("not sure" in r for r in replies), \
        "a denial must never come back as uncertainty"
    assert not session.history.not_known, "a denial is an answer, not a gap"


def test_extraction_is_told_which_question_it_answers():
    """"no i dont" answers whichever field was just asked about. Handed over
    alone it answers nothing, and the patient gets asked again."""
    seen = {}

    class Recording(Quiet):
        def extract(self, message, slots, asked="", answering=""):
            seen.update({"asked": asked, "answering": answering})
            return super().extract(message, slots, asked, answering)

    session = IntakeSession(_provider=Recording())
    session.send("I have a cough")
    session.send("no")

    assert seen["asked"], "the question just asked must travel with the reply"
    assert seen["answering"], "so must the field it belongs to"
