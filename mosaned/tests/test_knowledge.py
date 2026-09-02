"""The information path, and the firewall that keeps it from diagnosing."""
from mosaned.domain import KnowledgeAnswer, MessageKind, SessionState, Source
from mosaned.engine import knowledge
from mosaned.engine.session import IntakeSession
from mosaned.providers.stub import StubProvider


class Grounded(StubProvider):
    """A provider that finds a real answer on a sanctioned domain."""
    def answer_question(self, question, domains):
        return KnowledgeAnswer(
            text="Ibuprofen can raise blood pressure and blunt some blood pressure medicines.",
            sources=[Source("Ibuprofen — NHS", "https://www.nhs.uk/medicines/ibuprofen")],
            grounded=True,
        )


class OffDomain(StubProvider):
    """A provider that answers, but cites something we never sanctioned."""
    def answer_question(self, question, domains):
        return KnowledgeAnswer(
            text="Some blog says it's completely fine.",
            sources=[Source("Random Health Blog", "https://totally-made-up-health.example/post")],
            grounded=True,
        )


class FoundNothing(StubProvider):
    def answer_question(self, question, domains):
        return KnowledgeAnswer(text="", sources=[], grounded=False)


class Recording(StubProvider):
    """Captures everything handed to the knowledge path."""
    seen: list = []

    def answer_question(self, question, domains):
        Recording.seen.append({"question": question, "domains": domains})
        return Grounded().answer_question(question, domains)


# ---- the firewall ----------------------------------------------------------

def test_the_question_path_never_receives_the_patients_history():
    """The load-bearing test. If the chart reaches this path, a helpful model
    tailors the answer -- and a tailored answer is a diagnosis."""
    Recording.seen = []
    session = IntakeSession(_provider=Recording())
    session.send("I've had a cough for 10 weeks and I've lost weight without trying")
    session.send("what causes a cough that goes on this long?")

    assert Recording.seen, "the knowledge path should have been called"
    handed_over = " ".join(
        str(v) for call in Recording.seen for v in call.values()
    ).lower()

    for leak in ("10 weeks", "lost weight", "cough for 10"):
        assert leak not in handed_over, f"history leaked into the question path: {leak!r}"


def test_answering_a_question_does_not_record_anything_about_the_patient():
    session = IntakeSession(_provider=Grounded())
    session.send("Is ibuprofen safe with blood pressure tablets?")
    assert session.history.hpi == {}
    assert session.history.background == {}
    assert not session.complaint_established


# ---- grounding -------------------------------------------------------------

def test_a_sanctioned_source_is_shown_with_its_citation():
    result = knowledge.answer("is ibuprofen ok?", Grounded())
    assert result.grounded
    rendered = knowledge.render(result)
    assert "NHS" in rendered


def test_a_source_we_never_sanctioned_is_refused():
    """A citation we did not sanction is not a citation."""
    result = knowledge.answer("is ibuprofen ok?", OffDomain())
    assert not result.grounded
    assert "made-up-health" not in result.text


def test_finding_nothing_declines_rather_than_improvising():
    result = knowledge.answer("what is a very obscure thing", FoundNothing())
    assert not result.grounded
    assert "don't have reliable information" in result.text


def test_a_provider_that_cannot_search_does_not_answer_from_memory():
    assert not StubProvider().answer_question("anything", ["nhs.uk"]).grounded


# ---- the fork --------------------------------------------------------------

def test_a_question_does_not_start_an_intake():
    session = IntakeSession(_provider=Grounded())
    result = session.send("Is ibuprofen safe with blood pressure tablets?")
    assert result["kind"] == "question"
    assert session.state is SessionState.GATHERING
    assert session.history.presenting_complaint_raw == ""


def test_a_question_mid_intake_answers_then_resumes():
    session = IntakeSession(_provider=Grounded())
    session.send("I've had a cough for 10 weeks")
    pending = session.pending_slot_id
    result = session.send("what causes a cough that goes on this long?")

    assert result["kind"] == "question"
    assert "general information" in result["reply"].lower()
    # The history it was part-way through is not derailed.
    assert result["awaiting"] == pending


def test_the_gate_runs_before_the_fork():
    """"My chest hurts, is that a heart attack?" is a question. It is also an
    emergency, and the gate must see it first."""
    session = IntakeSession(_provider=Grounded())
    result = session.send("I have crushing chest pain, is that a heart attack?")
    assert session.state is SessionState.ESCALATED
    assert result.get("kind") != "question"


def test_stub_fork_reads_intent():
    stub = StubProvider()
    assert stub.classify_intent("I've had a cough for 3 days") is MessageKind.SYMPTOM
    assert stub.classify_intent("What is a chronic cough?") is MessageKind.QUESTION


def test_the_fork_costs_no_extra_model_call():
    """Intent is read in the same pass as the gate. A separate call for it was
    a third of the per-message quota spent re-reading the same message.

    The double answers `assess` without delegating, and makes `classify_intent`
    fatal -- so the session reaching for it fails loudly rather than quietly
    costing a request.
    """
    from mosaned.domain import FreeConcern, GateAssessment

    calls = {"assess": 0}

    class OneReadOnly(Grounded):
        def assess(self, message, flags, context=""):
            calls["assess"] += 1
            asking = "?" in message
            return GateAssessment(
                present_flag_ids=[],
                free_concern=FreeConcern(concerned=False),
                kind=MessageKind.QUESTION if asking else MessageKind.SYMPTOM,
            )

        def classify_intent(self, message):
            raise AssertionError("the fork must not cost its own model call")

    session = IntakeSession(_provider=OneReadOnly())
    session.send("I've had a cough for 3 days")
    result = session.send("Is ibuprofen safe with blood pressure tablets?")

    assert calls["assess"] == 2, "one read per message, no more"
    assert result["kind"] == "question", "the merged read still drives the fork"
