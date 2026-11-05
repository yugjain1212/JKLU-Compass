"""Pydantic schemas for the JKLU Compass API."""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request body for asking JKLU Compass a question."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question about JKLU.",
        examples=["What B.Tech specialisations are available at JKLU?"],
    )

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
