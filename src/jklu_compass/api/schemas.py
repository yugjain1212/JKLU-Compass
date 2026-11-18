"""Pydantic schemas for the JKLU Compass API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ConversationTurn(BaseModel):
    """One recent message used only to resolve chat follow-ups."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Conversation content must not be empty.")
        return value


class QueryRequest(BaseModel):
    """Request body for asking JKLU Compass a question."""

    question: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="Question about JKLU.",
        examples=["What B.Tech specialisations are available at JKLU?"],
    )
    history: list[ConversationTurn] = Field(default_factory=list, max_length=8)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=100)
    clarification_mode: Literal["adaptive", "always"] = "adaptive"
    clarification_id: str | None = Field(default=None, min_length=1, max_length=100)
    option_id: str | None = Field(default=None, min_length=1, max_length=100)
    custom_answer: str | None = Field(default=None, max_length=500)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str | None) -> str | None:
        """Reject whitespace-only questions and normalize surrounding whitespace."""
        if value is None:
            return None
        value = value.strip()

        if not value:
            raise ValueError("Question must not be empty.")

        return value

    @field_validator("custom_answer")
    @classmethod
    def validate_custom_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Custom clarification must not be empty.")
        return value

    @model_validator(mode="after")
    def validate_action(self) -> "QueryRequest":
        if self.clarification_id:
            if not self.option_id:
                raise ValueError("option_id is required when selecting a clarification.")
            if not self.conversation_id:
                raise ValueError("conversation_id is required when selecting a clarification.")
            if self.option_id == "other" and not self.custom_answer:
                raise ValueError("custom_answer is required for the Other option.")
            if self.option_id != "other" and self.custom_answer:
                raise ValueError("custom_answer is only valid for the Other option.")
        elif self.option_id:
            raise ValueError("clarification_id is required when selecting an option.")
        elif self.custom_answer:
            raise ValueError("custom_answer is only valid when selecting Other.")
        elif not self.question:
            raise ValueError("question is required when starting a query.")
        return self


class SourceResponse(BaseModel):
    """Citation metadata returned with an answer."""

    number: int
    title: str
    url: str | None
    chunk_id: str


class StructuredProgrammeResponse(BaseModel):
    """One programme and its supporting citation number."""

    name: str
    source_number: int


class StructuredAnswerResponse(BaseModel):
    """Machine-readable answer data."""

    answer_type: str
    programmes: list[StructuredProgrammeResponse]


class ClarificationOptionResponse(BaseModel):
    """One selectable clarification option."""

    id: str
    question: str


class QueryResponse(BaseModel):
    """Answer or clarification response."""

    status: Literal["answer", "clarification_required"]
    question: str | None = None
    answer: str | None = None
    sources: list[SourceResponse] = Field(default_factory=list)
    retrieved_chunks: int = 0
    structured_data: StructuredAnswerResponse | None = None
    conversation_id: str
    clarification_id: str | None = None
    clarification_message: str | None = None
    options: list[ClarificationOptionResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    pipeline: str


class ErrorResponse(BaseModel):
    """Standard API error response."""

    detail: str
