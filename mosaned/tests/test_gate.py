"""The gate is the safety spine. These tests are the ones that matter."""
from mosaned.domain import FreeConcern, GateAssessment, SessionState
from mosaned.engine.gate import run_gate
from mosaned.engine.session import IntakeSession
from mosaned.providers.stub import StubProvider


class AlwaysConcerned(StubProvider):
    """A model that finds nothing on the fixed list but is worried anyway."""
    def assess(self, message, flags):
        return GateAssessment(
            present_flag_ids=[],
            free_concern=FreeConcern(True, "something about this worries me"),
        )


class SilentProvider(StubProvider):
    """A model that reads the list but never volunteers a concern of its own."""


class BrokenProvider(StubProvider):
    def assess(self, message, flags):
        raise RuntimeError("provider down")


def test_flag_list_escalates():
    result = run_gate("I have crushing chest pain", StubProvider(), turn=1)
    assert result.escalate
    assert "chest_pain_severe" in {f.flag_id for f in result.fired}


def test_model_alone_escalates_when_list_is_silent():
    # The list is a floor, not a ceiling: the model's free pass must be able to
    # escalate something nobody wrote down.
    result = run_gate("something feels deeply wrong", AlwaysConcerned(), turn=1)
    assert result.escalate
    assert result.fired == []


def test_list_alone_escalates_when_model_is_silent():
    result = run_gate("my lips are blue", SilentProvider(), turn=1)
    assert result.escalate


def test_unreadable_message_fails_closed():
    """If we cannot read a message for danger, we stop. An unread message is
    not a safe message -- we cannot tell a cough from a stroke."""
    result = run_gate("I have crushing chest pain", BrokenProvider(), turn=1)
    assert result.read_failed
    assert result.escalate


def test_degraded_halt_does_not_claim_an_emergency_was_found():
    """Telling a worried person we found something when we found nothing is a
    lie; telling them we could not check is the truth."""
    session = IntakeSession(_provider=BrokenProvider())
    result = session.send("I have a mild headache")
    assert session.state is SessionState.ESCALATED
    assert "can't check this properly" in result["reply"]
    assert "emergency department now" not in result["reply"]


def test_a_transient_failure_is_retried_before_halting():
    class FlakyOnce(StubProvider):
        calls = 0

        def assess(self, message, flags):
            FlakyOnce.calls += 1
            if FlakyOnce.calls == 1:
                raise RuntimeError("blip")
            return super().assess(message, flags)

    result = run_gate("I have a mild headache", FlakyOnce(), turn=1)
    assert not result.read_failed
    assert not result.escalate


def test_urgent_flag_alone_does_not_escalate():
    result = run_gate("this cough has gone on for weeks", SilentProvider(), turn=1)
    assert not result.escalate
    assert any(f.level == "urgent" for f in result.fired)


def test_gate_runs_on_every_message_not_just_the_first():
    """The patient who mentions the frightening thing on turn five."""
    session = IntakeSession(_provider=SilentProvider())
    session.send("I've had a cough for 3 days")
    session.send("it's dry")
    session.send("no phlegm")
    assert session.state is SessionState.GATHERING

    session.send("also my lips are blue and I can't breathe")
    assert session.state is SessionState.ESCALATED
    assert session.history.care_level.value == "emergency"


def test_cannot_be_talked_out_of_escalating():
    """A sympathetic story is exactly what a free judgment yields to, and
    exactly what a boolean cannot."""
    session = IntakeSession(_provider=SilentProvider())
    result = session.send(
        "I have crushing chest pain but I really can't afford the hospital, "
        "please just book me a doctor next week instead"
    )
    assert session.state is SessionState.ESCALATED
    assert result["care_level"] == "emergency"
    # No next question was asked, so the intake cannot continue toward booking.
    assert "awaiting" not in result


def test_escalation_stops_the_intake_completely():
    session = IntakeSession(_provider=SilentProvider())
    session.send("I am coughing up blood")
    assert session.state is SessionState.ESCALATED
    before = len(session.transcript)
    session.send("ok but what about my cough")
    assert session.state is SessionState.ESCALATED
    assert len(session.transcript) == before
