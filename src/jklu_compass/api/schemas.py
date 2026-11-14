"""Pydantic schemas for the JKLU Compass API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


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

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question about JKLU.",
        examples=["What B.Tech specialisations are available at JKLU?"],
    )
    history: list[ConversationTurn] = Field(default_factory=list, max_length=8)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject whitespace-only questions and normalize surrounding whitespace."""
        value = value.strip()

        if not value:
            raise ValueError("Question must not be empty.")

        return value


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


class QueryResponse(BaseModel):
    """Complete RAG response."""

    question: str
    answer: str
    sources: list[SourceResponse]
    retrieved_chunks: int
    structured_data: StructuredAnswerResponse | None = None


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    pipeline: str


class ErrorResponse(BaseModel):
    """Standard API error response."""

    detail: str
