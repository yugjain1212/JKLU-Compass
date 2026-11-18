"""Tests for the pre-retrieval query clarification layer."""

from jklu_compass.rag.query_clarifier import QueryClarifier


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""

    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        self.prompt = prompt
        return self.response


def test_ambiguous_question_returns_selectable_interpretations() -> None:
    llm = FakeLLM(
        '{"clarification_required":true,"message":"Which dean do you mean?",'
        '"options":["The current IET dean","The Dean of Academic Affairs"]}'
    )
    decision = QueryClarifier(llm).clarify("Who is the dean IET?")
    assert decision.required
    assert len(decision.options) == 2
    assert decision.options[0].question == "The current IET dean"


def test_broad_question_can_offer_scope_refinements() -> None:
    llm = FakeLLM(
        '{"clarification_required":true,"message":"What would you like to know about JKLU?",'
        '"options":["Academic programmes","Admissions and fees","Campus life"]}'
    )
    decision = QueryClarifier(llm).clarify("Tell me about JKLU")
    assert decision.required
    assert len(decision.options) == 3


def test_specific_question_proceeds_without_clarification() -> None:
    llm = FakeLLM('{"clarification_required":false,"message":"","options":[]}')
    decision = QueryClarifier(llm).clarify("What is the annual B.Tech academic fee?")
    assert not decision.required
    assert decision.options == ()


def test_follow_up_history_is_available_for_reference_resolution() -> None:
    llm = FakeLLM('{"clarification_required":false,"message":"","options":[]}')
    QueryClarifier(llm).clarify(
        "What are his qualifications?",
        (("user", "Who is the IET dean?"), ("assistant", "Dr. Tapas Kumar is the dean.")),
    )
    assert "Dr. Tapas Kumar" in llm.prompt


def test_invalid_json_falls_back_to_direct_retrieval_in_adaptive_mode() -> None:
    decision = QueryClarifier(FakeLLM("not json")).clarify("Who is the IET dean?")
    assert not decision.required


def test_invalid_json_in_always_mode_uses_safe_refinements() -> None:
    decision = QueryClarifier(FakeLLM("not json")).clarify("Tell me about JKLU", mode="always")
    assert decision.required
    assert 2 <= len(decision.options) <= 4
