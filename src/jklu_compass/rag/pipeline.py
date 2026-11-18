"""Retrieval-augmented generation pipeline for JKLU Compass."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from html import unescape
from urllib.parse import urlparse

import httpx

from jklu_compass.generation.local_llm import LocalLLM
from jklu_compass.rag.structured_extractor import (
    StructuredExtractor,
    StructuredItem,
)
from jklu_compass.retrieval.retriever import DocumentRetriever, RetrievalResult

DEGREE_PATTERNS = {
    "btech": re.compile(r"\b(?:b\.?\s*tech|bachelor\s+of\s+technology)\b", re.IGNORECASE),
    "bdes": re.compile(r"\b(?:b\.?\s*des|bachelor\s+of\s+design)\b", re.IGNORECASE),
    "mtech": re.compile(r"\b(?:m\.?\s*tech|master\s+of\s+technology)\b", re.IGNORECASE),
    "mdes": re.compile(r"\b(?:m\.?\s*des|master\s+of\s+design)\b", re.IGNORECASE),
    "bba": re.compile(r"\b(?:bba|bachelor\s+of\s+business(?:\s+administration)?)\b", re.IGNORECASE),
    "mba": re.compile(r"\b(?:mba|master\s+of\s+business(?:\s+administration)?)\b", re.IGNORECASE),
}
FEE_AMOUNT_PATTERNS = {
    "academic": re.compile(
        r"\bacademic\s+fee(?:\s*\(annual\))?\s*:\s*\*{0,2}\s*(?:-\s*)?(?:INR|₹)?\s*([\d,]+)",
        re.IGNORECASE,
    ),
    "technology": re.compile(
        r"\btechnology\s+fee(?:\s*\(annual\))?\s*:\s*\*{0,2}\s*(?:-\s*)?(?:INR|₹)?\s*([\d,]+)",
        re.IGNORECASE,
    ),
}
GREETING_PATTERNS = (
    re.compile(r"^(?:hi|hello|hey)(?: there)?$"),
    re.compile(r"^good (?:morning|afternoon|evening)$"),
    re.compile(r"^namaste$"),
)
HELP_PATTERNS = (
    re.compile(r"^(?:(?:hi|hello|hey) )?how can you help(?: me)?$"),
    re.compile(r"^(?:(?:hi|hello|hey) )?what can you help(?: me)? with$"),
    re.compile(r"^(?:(?:hi|hello|hey) )?what can i ask(?: you)?$"),
    re.compile(r"^(?:(?:hi|hello|hey) )?what do you do$"),
)
FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:they|them|their|he|him|his|she|her|it|its|this|that|these|those)\b"
    r"|\bthere\s+(?:qualifications?|education|degrees?)\b"
    r"|\b(?:what about|how about|tell me more|more about|and also|and what|and how)\b",
    re.IGNORECASE,
)
CURRENT_VC_QUERY = re.compile(
    r"\b(?:current|currently|now|today|present(?:ly)?)\b.{0,80}"
    r"\b(?:vice[\s-]?chancellor|v\.?\s*c\.?)\b"
    r"|\b(?:vice[\s-]?chancellor|v\.?\s*c\.?)\b.{0,80}"
    r"\b(?:current|currently|now|today|present(?:ly)?)\b"
    r"|\bwho\s+(?:is|are)\s+(?:(?:the|jklu's?)\s+)?"
    r"(?:vice[\s-]?chancellor|v\.?\s*c\.?)\b",
    re.IGNORECASE,
)
LIVE_VC_NAME_PATTERN = re.compile(
    r"(?P<name>(?:Professor|Prof\.?|Dr\.?)\s+"
    r"[A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4})"
    r"\s+(?:\|\s*)?Vice[\s-]?Chancellor\b"
)
LIVE_VC_URL = "https://jklu.edu.in/about-us/leadership-governance"
LIVE_VC_TITLE = "Leadership & Governance | JK Lakshmipat University"
INDEXED_VC_URLS = frozenset(
    {
        LIVE_VC_URL,
        "https://jklu.edu.in/about-us/leadership-insights/vc-message",
    }
)
IET_SOURCE_URL = "https://jklu.edu.in/institute/engineering-technology"
IET_DEAN_EVIDENCE_PATTERN = re.compile(
    r"(?m)^\s*#{1,6}\s*(?P<name>(?:Dr\.?|Prof\.?)\s+"
    r"[A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4})\s*\n+"
    r"\s*Dean\s*\n+\s*Engineering\s*(?:&|and)\s*Technology\b",
    re.IGNORECASE,
)
QUESTION_PART_SEPARATOR = re.compile(
    r"\s+(?:and\s+(?:also\s+)?|also\s+)"
    r"(?=(?:tell|show|give|explain|what|when|where|which|who|can|could|do|is|are)\b)",
    re.IGNORECASE,
)
CLASS_TIMING_PATTERN = re.compile(
    r"\b(?:class(?:es)?\s+(?:tim(?:e|ing|ings)|schedule|timetable)"
    r"|tim(?:e|ing|ings)\s+of\s+(?:jklu\s+)?classes)\b",
    re.IGNORECASE,
)
BTECH_YEAR_SEMESTER_PATTERN = re.compile(
    r"\b(?:b\.?\s*tech|bachelor\s+of\s+technology)\b.{0,60}?"
    r"(?P<year>[1-4])(?:st|nd|rd|th)?\s+year\b"
    r"(?:\s*(?:,|and|of)\s*)?"
    r"(?P<semester>[1-8])(?:st|nd|rd|th)?\s+sem(?:ester)?\b",
    re.IGNORECASE,
)
TITLED_PERSON_NAME_PATTERN = re.compile(
    r"\b(?:(?i:assistant|associate|adjunct|visiting|asst\.?|assoc\.?)\s+)?"
    r"(?i:professor|prof\.?|dr\.?|dean|director)\s+(?:(?i:dr\.?|prof\.?)\s+)?"
    r"(?P<name>[A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4})\b",
    re.IGNORECASE,
)
UNTITLED_PERSON_NAME_PATTERN = re.compile(
    r"\b(?:about|of|for|from|on|does|did|is|are|was|were|has|have)\s+"
    r"(?P<name>[A-Za-z][A-Za-z'’.-]+(?:\s+[A-Za-z][A-Za-z'’.-]+){1,3})\b",
    re.IGNORECASE,
)
BARE_PERSON_NAME_PATTERN = re.compile(
    r"^\s*(?:(?:dr\.?|prof\.?|professor|assistant\s+professor|asst\.?\s+prof\.?)\s+)?"
    r"(?P<name>[a-z][a-z'’.-]+(?:\s+[a-z][a-z'’.-]+){1,3})[?.!]*\s*$",
    re.IGNORECASE,
)
NON_PERSON_NAME_TERMS = frozenset(
    {
        "a",
        "about",
        "an",
        "are",
        "assistant",
        "can",
        "could",
        "college",
        "current",
        "dean",
        "degree",
        "degrees",
        "details",
        "dr",
        "education",
        "engineering",
        "faculty",
        "he",
        "her",
        "his",
        "institute",
        "is",
        "it",
        "its",
        "jklu",
        "of",
        "professor",
        "profile",
        "prof",
        "qualification",
        "qualifications",
        "school",
        "she",
        "tell",
        "technology",
        "the",
        "their",
        "them",
        "they",
        "this",
        "that",
        "university",
        "vice",
        "was",
        "were",
        "what",
        "who",
        "chancellor",
        "you",
    }
)
PERSON_EDUCATION_QUERY_PATTERN = re.compile(
    r"\b(?:education|educational|qualifications?|degrees?|alma\s+mater|studied|graduated)\b",
    re.IGNORECASE,
)
PERSON_EDUCATION_EVIDENCE_PATTERN = re.compile(
    r"\b(?:education|academic\s+qualifications?|degrees?)\s*[:\-]\s*\S"
    r"|\b(?:holds?|earned|received|completed|graduated|studied)\b.{0,140}"
    r"\b(?:bachelor|b\.?\s*tech|b\.?\s*des|master|m\.?\s*tech|m\.?\s*des|"
    r"ph\.?\s*d|doctorate|degree|diploma)\b"
    r"|\b(?:bachelor|b\.?\s*tech|b\.?\s*des|master|m\.?\s*tech|m\.?\s*des|"
    r"ph\.?\s*d|doctorate|diploma)\b.{0,100}\b(?:from|in|at)\b",
    re.IGNORECASE,
)
PERSON_EDUCATION_SECTION_PATTERN = re.compile(
    r"^(?:education(?:al\s+background)?|academic\s+qualifications?|qualifications?|degrees?)$",
    re.IGNORECASE,
)
PERSON_CREDENTIAL_PATTERN = re.compile(
    r"\b(?:bachelor|b\.?\s*tech|b\.?\s*des|master|m\.?\s*tech|m\.?\s*des|"
    r"ph\.?\s*d|doctorate|diploma)\b",
    re.IGNORECASE,
)
DEGREE_LABELS = {
    "btech": "B.Tech",
    "bdes": "B.Des",
    "mtech": "M.Tech",
    "mdes": "M.Des",
    "bba": "BBA",
    "mba": "MBA",
}


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """A source used to construct a RAG answer."""

    number: int
    title: str
    url: str | None
    chunk_id: str


@dataclass(frozen=True, slots=True)
class StructuredProgramme:
    """A B.Tech programme with its supporting source citation."""

    name: str
    source_number: int


@dataclass(frozen=True, slots=True)
class StructuredAnswer:
    """Machine-readable data extracted from retrieved evidence."""

    answer_type: str
    programmes: tuple[StructuredProgramme, ...]


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """The generated answer and its supporting sources."""

    question: str
    answer: str
    sources: tuple[SourceCitation, ...]
    retrieved_chunks: int
    structured_data: StructuredAnswer | None = None


class RAGPipeline:
    """Retrieve university documents and generate grounded answers."""

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        llm: LocalLLM | None = None,
        top_k: int = 8,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.retriever = retriever or DocumentRetriever()
        self.llm = llm or LocalLLM()
        self.top_k = top_k

    def ask(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> RAGResponse:
        """Answer a question using retrieved university documents."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        question = question.strip()
        history = tuple(history[-8:])

        small_talk_answer = self._small_talk_answer(question)
        if small_talk_answer is not None:
            return RAGResponse(
                question=question,
                answer=small_talk_answer,
                sources=(),
                retrieved_chunks=0,
            )

        name_lookup = self._bare_person_name_query(question)
        if name_lookup and isinstance(self.retriever, DocumentRetriever):
            profile_results = self.retriever.retrieve_faculty_profile(name_lookup, top_k=1)
            _, profile_citations = self._build_context(profile_results)
            if profile_citations:
                citation = profile_citations[0]
                profile_name = citation.title.split(",", maxsplit=1)[0]
                return RAGResponse(
                    question=question,
                    answer=(
                        f"I found the JKLU faculty profile for {profile_name} "
                        f"[Source {citation.number}]. What would you like to know?"
                    ),
                    sources=(citation,),
                    retrieved_chunks=len(profile_results),
                )

        if self._is_iet_dean_query(question):
            return self._answer_iet_dean(question)

        if CURRENT_VC_QUERY.search(question):
            return self._answer_current_vc(question)

        if self._requires_reference(question) and not self._subject_history(history):
            return RAGResponse(
                question=question,
                answer=(
                    "Who are you asking about? Please include the person's name or the "
                    "programme so I can look up the right JKLU source."
                ),
                sources=(),
                retrieved_chunks=0,
            )

        retrieval_queries = self._build_retrieval_queries(question, history)
        retrieval_limit = (
            max(self.top_k, 6)
            if DocumentRetriever.is_btech_programme_list_query(question)
            else self.top_k
        )
        results_by_query = [
            self.retriever.retrieve(retrieval_query, top_k=retrieval_limit)
            for retrieval_query in retrieval_queries
        ]
        results = self._merge_retrieval_results(results_by_query, retrieval_limit)
        context_results = (
            results
            if len(retrieval_queries) > 1
            else self._scope_results_to_question(question, results)
        )
        requested_people = self._requested_person_names(question, history)
        if requested_people:
            context_results = self._scope_results_to_people(context_results, requested_people)
            if len(requested_people) == 1 and isinstance(self.retriever, DocumentRetriever):
                profile_results = self.retriever.retrieve_faculty_profile(
                    requested_people[0],
                    top_k=10,
                )
                if profile_results:
                    context_results = profile_results

        structured_items = (
            StructuredExtractor.extract_btech_programmes(context_results)
            if DocumentRetriever.is_btech_programme_list_query(question)
            else []
        )

        structured_context = self._build_structured_context(structured_items)

        context, citations = self._build_context(context_results)

        if not citations:
            return RAGResponse(
                question=question,
                answer=(
                    "I couldn't find usable information in the indexed "
                    "JKLU documents to answer that question. Please try "
                    "rephrasing your question or consult the official "
                    "JKLU website."
                ),
                sources=(),
                retrieved_chunks=len(results),
            )

        person_education_answer = self._build_person_education_answer(
            question,
            context_results,
            citations,
            requested_people,
        )
        if person_education_answer is not None:
            answer, education_citations = person_education_answer
            return RAGResponse(
                question=question,
                answer=answer,
                sources=education_citations,
                retrieved_chunks=len(results),
            )

        person_section_answer = self._build_person_section_answer(
            context_results,
            citations,
            requested_people,
        )
        if person_section_answer is not None:
            answer, person_citations = person_section_answer
            return RAGResponse(
                question=question,
                answer=answer,
                sources=person_citations,
                retrieved_chunks=len(context_results),
            )

        # Use deterministic rendering for verified B.Tech programme lists.
        structured_data = self._build_structured_answer(
            structured_items,
            citations,
        )

        if structured_data is not None:
            answer = self._render_structured_answer(structured_data)

            return RAGResponse(
                question=question,
                answer=answer,
                sources=tuple(citations),
                retrieved_chunks=len(results),
                structured_data=structured_data,
            )

        fee_answer = self._build_structured_fee_answer(question, context_results, citations)
        if fee_answer is not None:
            answer, fee_citations = fee_answer
            return RAGResponse(
                question=question,
                answer=answer,
                sources=fee_citations,
                retrieved_chunks=len(results),
            )

        prompt = self._build_prompt(
            question,
            context,
            structured_context=structured_context,
            history=history if self._is_follow_up_question(question) else (),
        )

        answer = self.llm.generate(
            prompt,
            temperature=0.1,
        )

        # Validate completeness for structured B.Tech questions.
        if structured_items:
            missing_items = self._find_missing_programmes(
                answer,
                structured_items,
            )

            # Retry generation once if any programme is missing.
            if missing_items:
                retry_prompt = self._build_completeness_retry_prompt(
                    question=question,
                    context=context,
                    structured_context=structured_context,
                    previous_answer=answer,
                    missing_items=missing_items,
                )

                answer = self.llm.generate(
                    retry_prompt,
                    temperature=0.1,
                )

                missing_items = self._find_missing_programmes(
                    answer,
                    structured_items,
                )

                # Deterministic fallback if the retry is still incomplete.
                if missing_items:
                    answer = self._append_missing_programmes(
                        answer,
                        missing_items,
                        citations,
                    )

        # Build machine-readable response data.
        structured_data = self._build_structured_answer(
            structured_items,
            citations,
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=tuple(citations),
            retrieved_chunks=len(results),
            structured_data=structured_data,
        )

    def is_indexed_faculty_query(self, question: str) -> bool:
        """Check whether a named faculty member has a matching indexed profile."""
        if not isinstance(self.retriever, DocumentRetriever):
            return False

        people = self._requested_person_names(question, ())
        return any(
            self.retriever.retrieve_faculty_profile(name, top_k=1)
            for name in people
        )

    def _answer_current_vc(self, question: str) -> RAGResponse:
        """Resolve current VC questions from JKLU's live leadership page."""
        citation = SourceCitation(
            number=1,
            title=LIVE_VC_TITLE,
            url=LIVE_VC_URL,
            chunk_id="live-jklu-leadership-governance",
        )
        name: str | None = None

        try:
            response = httpx.get(
                LIVE_VC_URL,
                follow_redirects=True,
                timeout=5.0,
                headers={"User-Agent": "JKLU Compass"},
            )
            response.raise_for_status()

            if response.url.host.casefold() in {"jklu.edu.in", "www.jklu.edu.in"}:
                page_html = re.sub(
                    r"(?is)<(script|style|noscript)\b[^>]*>.*?</\1\s*>",
                    " ",
                    response.text,
                )
                page_text = unescape(re.sub(r"(?s)<[^>]+>", " ", page_html))
                page_text = re.sub(r"\s+", " ", page_text)
                match = LIVE_VC_NAME_PATTERN.search(page_text)
                if match:
                    name = match.group("name")
        except httpx.HTTPError:
            pass

        if name is None:
            name, citation = self._indexed_current_vc()

        if name:
            answer = f"JKLU's current Vice-Chancellor is {name} [Source 1]."
            retrieved_chunks = 1
        else:
            answer = (
                "I couldn't verify JKLU's current Vice-Chancellor from its live leadership page. "
                "Please check [Source 1]."
            )
            retrieved_chunks = 0

        return RAGResponse(
            question=question,
            answer=answer,
            sources=(citation,),
            retrieved_chunks=retrieved_chunks,
        )

    def _answer_iet_dean(self, question: str) -> RAGResponse:
        """Answer IET dean questions only from the IET's own leadership block."""
        try:
            results = self.retriever.retrieve(
                "Dr. Tapas Kumar Dean Engineering & Technology IET",
                top_k=max(self.top_k, 12),
            )
        except (OSError, RuntimeError, ValueError):
            results = []

        iet_path = urlparse(IET_SOURCE_URL).path
        for result in results:
            if not result.source_url or urlparse(result.source_url).path.rstrip("/") != iet_path:
                continue
            match = IET_DEAN_EVIDENCE_PATTERN.search(result.content)
            if match is None:
                continue

            _, citations = self._build_context([result])
            if not citations:
                continue
            citation = citations[0]
            return RAGResponse(
                question=question,
                answer=(
                    f"The IET page lists {match.group('name')} as Dean of Engineering & "
                    f"Technology [Source 1]."
                ),
                sources=(citation,),
                retrieved_chunks=1,
            )

        return RAGResponse(
            question=question,
            answer=(
                "I couldn't confirm the IET dean from the indexed IET leadership page. "
                "Please check the official page for the current listing."
            ),
            sources=(),
            retrieved_chunks=0,
        )

    @staticmethod
    def _is_iet_dean_query(question: str) -> bool:
        return bool(
            re.search(r"\bdean\b", question, re.IGNORECASE)
            and re.search(
                r"\b(?:iet|institute\s+of\s+engineering(?:\s+and|\s*&)?\s+technology)\b",
                question,
                re.IGNORECASE,
            )
        )

    def _indexed_current_vc(self) -> tuple[str | None, SourceCitation]:
        """Use the indexed official VC page when Cloudflare blocks the live request."""
        fallback_citation = SourceCitation(
            number=1,
            title=LIVE_VC_TITLE,
            url=LIVE_VC_URL,
            chunk_id="live-jklu-leadership-governance",
        )

        try:
            results = self.retriever.retrieve(
                "current Vice-Chancellor JKLU leadership",
                top_k=self.top_k,
            )
        except (OSError, RuntimeError, ValueError):
            return None, fallback_citation

        for result in results:
            if result.source_url not in INDEXED_VC_URLS:
                continue

            match = LIVE_VC_NAME_PATTERN.search(result.content)
            if match is None:
                continue

            return (
                match.group("name"),
                SourceCitation(
                    number=1,
                    title=result.title,
                    url=result.source_url,
                    chunk_id=result.id,
                ),
            )

        return None, fallback_citation

    @staticmethod
    def _small_talk_answer(question: str) -> str | None:
        """Handle greetings and capability questions without document retrieval."""
        normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()

        if any(pattern.fullmatch(normalized) for pattern in HELP_PATTERNS):
            return (
                "Hi! I'm JKLU Compass. I can help with JKLU programmes, admissions, "
                "fees, faculty, campus facilities, policies, and placements using "
                "the indexed university sources. What would you like to know?"
            )

        if any(pattern.fullmatch(normalized) for pattern in GREETING_PATTERNS):
            return "Hi! I'm JKLU Compass. What would you like to know about JKLU?"

        if normalized in {"thanks", "thank you", "thx"}:
            return "You're welcome. What else would you like to know about JKLU?"

        return None

    @staticmethod
    def _is_follow_up_question(question: str) -> bool:
        return FOLLOW_UP_PATTERN.search(question) is not None

    @staticmethod
    def _requires_reference(question: str) -> bool:
        return bool(
            re.search(
                r"\b(?:they|them|their|he|him|his|she|her|it|its|this|that|these|those)\b"
                r"|\bthere\s+(?:qualifications?|education|degrees?)\b",
                question,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _bare_person_name_query(question: str) -> str | None:
        """Return a short name-only query for exact indexed faculty lookup."""
        match = BARE_PERSON_NAME_PATTERN.fullmatch(question)
        return match.group("name") if match else None

    @staticmethod
    def _subject_history(
        history: Sequence[tuple[str, str]],
    ) -> tuple[tuple[str, str], ...]:
        """Return the latest user question that introduces a concrete subject."""
        generic_words = {
            "a",
            "about",
            "also",
            "am",
            "and",
            "are",
            "can",
            "could",
            "do",
            "does",
            "for",
            "he",
            "her",
            "him",
            "his",
            "how",
            "i",
            "it",
            "me",
            "more",
            "my",
            "of",
            "please",
            "she",
            "tell",
            "that",
            "the",
            "their",
            "them",
            "there",
            "these",
            "they",
            "this",
            "those",
            "to",
            "was",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
            "would",
            "you",
            "your",
            "name",
            "one",
            "person",
            "faculty",
        }
        follow_up_words = {
            "degree",
            "degrees",
            "education",
            "qualification",
            "qualifications",
        }

        for index in range(len(history) - 1, -1, -1):
            role, content = history[index]
            if role != "user" or RAGPipeline._small_talk_answer(content) is not None:
                continue

            words = set(re.findall(r"[a-z0-9]+", content.casefold()))
            if words - generic_words - follow_up_words:
                end = index + 1
                if end < len(history) and history[end][0] == "assistant":
                    end += 1
                return tuple(history[index:end])

        return ()

    @staticmethod
    def _build_retrieval_query(
        question: str,
        history: Sequence[tuple[str, str]],
    ) -> str:
        """Add the latest dialogue context only when the question refers back."""
        if not RAGPipeline._is_follow_up_question(question):
            return question

        subject_turns = RAGPipeline._subject_history(history)
        if not subject_turns:
            return question

        recent_context = "\n".join(f"{role}: {content[:500]}" for role, content in subject_turns)
        return f"{question}\nRecent conversation for reference resolution:\n{recent_context}"

    @staticmethod
    def _split_question_parts(question: str) -> list[str]:
        """Split explicit additional requests without splitting names or course titles."""
        parts = [part.strip(" ,;") for part in QUESTION_PART_SEPARATOR.split(question)]
        return [part for part in parts if part]

    @classmethod
    def _build_retrieval_queries(
        cls,
        question: str,
        history: Sequence[tuple[str, str]],
    ) -> list[str]:
        """Build focused searches for each clearly separate request."""
        queries = []
        for part in cls._split_question_parts(question):
            query = cls._build_retrieval_query(part, history)
            expansions = []

            # Leadership tables often contain many unrelated "Dean" entries.
            # Search the IET's own faculty section explicitly so that a dean of
            # another institute (or a university-wide role) cannot dominate.
            asks_iet_dean = bool(
                re.search(r"\b(?:dean|director)\b", part, re.I)
                and re.search(
                    r"\b(?:iet|institute\s+of\s+engineering(?:\s+and|\s*&)?\s+technology)\b",
                    part,
                    re.I,
                )
            )
            if asks_iet_dean:
                queries.append(
                    "JKLU Institute of Engineering & Technology IET faculty Dr. Tapas Kumar Dean"
                )

            if CLASS_TIMING_PATTERN.search(part):
                expansions.append(
                    "daily class timetable regular lecture hours daily start time end time"
                )

            semester_match = BTECH_YEAR_SEMESTER_PATTERN.search(part)
            if semester_match:
                year = int(semester_match.group("year"))
                semester = int(semester_match.group("semester"))
                ordinals = {
                    1: "first",
                    2: "second",
                    3: "third",
                    4: "fourth",
                    5: "fifth",
                    6: "sixth",
                    7: "seventh",
                    8: "eighth",
                }
                roman_numerals = {
                    1: "I",
                    2: "II",
                    3: "III",
                    4: "IV",
                    5: "V",
                    6: "VI",
                    7: "VII",
                    8: "VIII",
                }
                expansions.append(
                    "B.Tech "
                    f"{ordinals[year]} year Semester {roman_numerals[semester]} "
                    f"{ordinals[semester]} semester curriculum syllabus subjects courses"
                )

            if expansions:
                query = f"{query}\nRelevant search terms: {'; '.join(expansions)}"
            queries.append(query)

        return queries or [question]

    @staticmethod
    def _merge_retrieval_results(
        result_groups: Sequence[list[RetrievalResult]],
        limit: int,
    ) -> list[RetrievalResult]:
        """Interleave results from separate question parts while deduplicating chunks."""
        merged = []
        seen_ids = set()
        max_group_size = max((len(group) for group in result_groups), default=0)

        for rank in range(max_group_size):
            for group in result_groups:
                if rank >= len(group):
                    continue
                result = group[rank]
                if result.id in seen_ids:
                    continue
                seen_ids.add(result.id)
                merged.append(result)
                if len(merged) >= limit:
                    return merged

        return merged

    @staticmethod
    def _build_structured_answer(
        items: list[StructuredItem],
        citations: list[SourceCitation],
    ) -> StructuredAnswer | None:
        """Build programme data with valid source references."""

        if not items:
            return None

        citation_by_chunk = {citation.chunk_id: citation.number for citation in citations}

        programmes = tuple(
            StructuredProgramme(
                name=item.name,
                source_number=citation_by_chunk[item.chunk_id],
            )
            for item in items
            if item.chunk_id in citation_by_chunk
        )

        if not programmes:
            return None

        return StructuredAnswer(
            answer_type="btech_programme_list",
            programmes=programmes,
        )

    @staticmethod
    def _render_structured_answer(data: StructuredAnswer) -> str:
        """Render a verified B.Tech programme list without LLM generation."""

        programme_lines = [
            f"{index}. {programme.name} [Source {programme.source_number}]"
            for index, programme in enumerate(data.programmes, start=1)
        ]

        programme_list = "\n".join(programme_lines)

        return (
            "The retrieved JKLU engineering brochure lists the following "
            "B.Tech programmes:\n\n"
            f"{programme_list}\n\n"
            "Note: This reflects the indexed brochure and does not "
            "independently confirm the current admissions catalogue."
        )

    @staticmethod
    def _find_missing_programmes(
        answer: str,
        items: list[StructuredItem],
    ) -> list[StructuredItem]:
        """Find programme names missing from the generated answer."""

        missing = []

        for item in items:
            # Avoid treating a specialised programme as the
            # general programme when its name is a prefix.
            pattern = re.escape(item.name) + r"(?!\s*\()"

            if not re.search(pattern, answer, flags=re.IGNORECASE):
                missing.append(item)

        return missing

    @staticmethod
    def _build_completeness_retry_prompt(
        question: str,
        context: str,
        structured_context: str,
        previous_answer: str,
        missing_items: list[StructuredItem],
    ) -> str:
        """Ask the LLM to correct an incomplete programme list."""

        missing_names = "\n".join(
            f"- {item.name} [Chunk ID: {item.chunk_id}]" for item in missing_items
        )

        return f"""You are correcting an incomplete answer for JKLU Compass.

The previous answer omitted one or more verified B.Tech programmes.

ORIGINAL QUESTION:
{question}

PREVIOUS ANSWER:
{previous_answer}

MISSING PROGRAMMES THAT MUST BE INCLUDED:
{missing_names}

VERIFIED STRUCTURED EVIDENCE:
{structured_context}

REFERENCE PASSAGES:
<context>
{context}
</context>

INSTRUCTIONS:
1. Produce a complete corrected answer to the original question.
2. Include every programme in the verified structured evidence.
3. Preserve the exact programme names.
4. Do not omit the general B.Tech programme.
5. Do not merge specialised programmes into the general programme.
6. Use only the supplied reference passages.
7. Cite each factual statement using the supplied [Source N] labels.
8. Do not invent programme descriptions or citations.
9. Keep the answer concise and use a bullet list.

Return the complete corrected answer, not just the missing items.

ANSWER:
"""

    @staticmethod
    def _append_missing_programmes(
        answer: str,
        missing_items: list[StructuredItem],
        citations: list[SourceCitation],
    ) -> str:
        """Append omitted programme names with their source citations."""

        citation_by_chunk = {citation.chunk_id: citation.number for citation in citations}

        additions = []

        for item in missing_items:
            source_number = citation_by_chunk.get(item.chunk_id)

            if source_number is None:
                continue

            additions.append(f"- {item.name} [Source {source_number}]")

        if not additions:
            return answer

        section = (
            "\n\n### Additional programmes supported by the "
            "retrieved documents\n\n" + "\n".join(additions)
        )

        return answer.rstrip() + section

    @classmethod
    def _scope_results_to_question(
        cls,
        question: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Prefer evidence for one named degree and the requested fee fields."""
        is_fee_question = bool(re.search(r"\b(?:fees?|tuition|costs?|charges?)\b", question, re.I))
        is_comparison = bool(
            re.search(r"\b(?:compare|comparison|versus|vs\.?|between|difference)\b", question, re.I)
        )
        asks_about_scholarship = bool(re.search(r"\b(?:scholarship|eligibility)\b", question, re.I))
        if is_comparison or (is_fee_question and asks_about_scholarship):
            return results

        question_degrees = cls._degree_mentions(question)
        if len(question_degrees) != 1:
            return results

        target_degree = next(iter(question_degrees))
        requested_fee_fields = {
            field
            for field in ("academic", "technology", "tuition", "hostel", "transport")
            if re.search(rf"\b{field}\s+fees?\b", question, re.I)
        }
        target_results: list[RetrievalResult] = []
        focused_results: list[RetrievalResult] = []
        fee_results: list[RetrievalResult] = []
        focused_fee_results: list[RetrievalResult] = []

        for result in results:
            searchable_text = " ".join(
                str(value)
                for value in (
                    result.content,
                    result.metadata.get("section_title", ""),
                    result.metadata.get("heading_path", ""),
                )
                if value
            )
            mentioned_degrees = cls._degree_mentions(searchable_text)

            if target_degree not in mentioned_degrees:
                continue

            target_results.append(result)
            searchable_lower = searchable_text.casefold()
            if not is_fee_question:
                has_requested_fee_field = False
            elif requested_fee_fields:
                has_requested_fee_field = any(
                    re.search(rf"\b{field}\s+fees?\b", searchable_lower)
                    for field in requested_fee_fields
                )
            else:
                has_requested_fee_field = bool(
                    re.search(
                        r"\b(?:academic|technology|tuition|hostel|transport|annual)\s+fees?\b"
                        r"|\bfee structure\b",
                        searchable_lower,
                    )
                )

            if has_requested_fee_field:
                if mentioned_degrees == {target_degree}:
                    fee_results.append(result)
                    focused_fee_results.append(result)
                else:
                    scoped_content = cls._filter_content_to_degree(
                        result.content,
                        target_degree,
                    )
                    if scoped_content and cls._has_requested_fee_field(
                        scoped_content,
                        requested_fee_fields,
                    ):
                        fee_results.append(replace(result, content=scoped_content))
            if mentioned_degrees == {target_degree}:
                focused_results.append(result)

        if is_fee_question:
            requested_years = cls._academic_years(question)
            year_specific_results = [
                result
                for result in fee_results
                if requested_years & cls._academic_years(result.content)
            ]
            if year_specific_results:
                return year_specific_results
            # Mixed tables are reduced to the target row; other fee topics are dropped.
            return (
                focused_fee_results or fee_results or focused_results or target_results or results
            )

        return focused_results or target_results or results

    @classmethod
    def _requested_person_names(
        cls,
        question: str,
        history: Sequence[tuple[str, str]],
    ) -> tuple[str, ...]:
        """Find explicitly named people in the question or its user-history subject."""
        names = []
        texts = [question]
        if cls._is_follow_up_question(question):
            texts.extend(
                content
                for role, content in cls._subject_history(history)
                if role == "user"
            )

        for text in texts:
            matches = [
                match.group("name")
                for pattern in (TITLED_PERSON_NAME_PATTERN, UNTITLED_PERSON_NAME_PATTERN)
                for match in pattern.finditer(text)
            ]
            for name in matches:
                normalized = cls._normalize_person_name(name)
                if not normalized or NON_PERSON_NAME_TERMS.intersection(normalized.split()):
                    continue
                if normalized not in {cls._normalize_person_name(item) for item in names}:
                    names.append(name)

        return tuple(names)

    @classmethod
    def _scope_results_to_people(
        cls,
        results: list[RetrievalResult],
        people: Sequence[str],
    ) -> list[RetrievalResult]:
        """Keep evidence tied to a person named by the user."""
        return [
            result
            for result in results
            if any(cls._result_matches_person(result, person) for person in people)
        ]

    @staticmethod
    def _normalize_person_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()

    @classmethod
    def _result_matches_person(cls, result: RetrievalResult, person: str) -> bool:
        requested_tokens = cls._normalize_person_name(person).split()
        if len(requested_tokens) < 2:
            return False

        faculty_slug = re.search(r"/faculty/([^/?#]+)", result.source_url or "", re.I)
        if faculty_slug:
            evidence_tokens = cls._normalize_person_name(
                faculty_slug.group(1).replace("-", " ")
            ).split()
            remaining_tokens = iter(evidence_tokens)
            return all(
                any(token == candidate for candidate in remaining_tokens)
                for token in requested_tokens
            )

        section_title = str(result.metadata.get("section_title", ""))
        section_person = TITLED_PERSON_NAME_PATTERN.search(section_title)
        if section_person:
            section_tokens = cls._normalize_person_name(section_person.group("name")).split()
            remaining_tokens = iter(section_tokens)
            return all(
                any(token == candidate for candidate in remaining_tokens)
                for token in requested_tokens
            )

        for segment in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", result.content):
            evidence_tokens = cls._normalize_person_name(segment).split()
            if cls._name_precedes_claim(requested_tokens, evidence_tokens):
                return True
        return False

    @staticmethod
    def _name_precedes_claim(name_tokens: Sequence[str], evidence_tokens: Sequence[str]) -> bool:
        """Require a named person to appear before other facts in the same sentence or row."""
        position = 0
        for token in name_tokens:
            while position < len(evidence_tokens) and evidence_tokens[position] != token:
                position += 1
            if position == len(evidence_tokens):
                return False
            position += 1
        return position < len(evidence_tokens)

    @staticmethod
    def _ordered_name_tokens_match(
        name_tokens: Sequence[str],
        evidence_tokens: Sequence[str],
    ) -> bool:
        position = 0
        for token in name_tokens:
            while position < len(evidence_tokens) and evidence_tokens[position] != token:
                position += 1
            if position == len(evidence_tokens):
                return False
            position += 1
        return True

    @classmethod
    def _build_person_education_answer(
        cls,
        question: str,
        results: list[RetrievalResult],
        citations: list[SourceCitation],
        people: Sequence[str],
    ) -> tuple[str, tuple[SourceCitation, ...]] | None:
        """Answer named-person education questions only with explicit profile evidence."""
        if not people or not PERSON_EDUCATION_QUERY_PATTERN.search(question):
            return None

        citation_by_chunk = {citation.chunk_id: citation for citation in citations}
        evidence = []
        for result in results:
            citation = citation_by_chunk.get(result.id)
            if citation is None:
                continue
            person = next(
                (name for name in people if cls._result_matches_person(result, name)),
                people[0],
            )
            segments = re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", result.content.strip())
            for index, segment in enumerate(segments):
                detail = segment.strip(" -*#\t")
                if detail and PERSON_EDUCATION_EVIDENCE_PATTERN.search(detail):
                    evidence.append((person, detail, citation))
                elif (
                    index > 0
                    and PERSON_EDUCATION_SECTION_PATTERN.fullmatch(
                        segments[index - 1].strip(" -*#\t")
                    )
                    and PERSON_CREDENTIAL_PATTERN.search(detail)
                ):
                    evidence.append((person, detail, citation))

        if not evidence:
            citation = next(
                (
                    citation_by_chunk[result.id]
                    for result in results
                    if result.id in citation_by_chunk
                ),
                None,
            )
            if citation is None:
                return None
            return (
                f"I couldn't confirm {people[0]}'s educational qualifications from the "
                f"matching indexed source [Source {citation.number}].",
                (citation,),
            )

        lines = []
        used_citations = []
        seen = set()
        for person, detail, citation in evidence:
            key = (person, detail.casefold(), citation.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {person}: {detail} [Source {citation.number}]")
            if citation not in used_citations:
                used_citations.append(citation)

        return "The matching indexed source states:\n" + "\n".join(lines), tuple(used_citations)

    @classmethod
    def _build_person_section_answer(
        cls,
        results: list[RetrievalResult],
        citations: list[SourceCitation],
        people: Sequence[str],
    ) -> tuple[str, tuple[SourceCitation, ...]] | None:
        """Use an exact person heading when no dedicated faculty URL is indexed."""
        if len(people) != 1:
            return None

        citation_by_chunk = {citation.chunk_id: citation for citation in citations}
        for result in results:
            if "/faculty/" in urlparse(result.source_url or "").path.casefold():
                continue

            section_person = TITLED_PERSON_NAME_PATTERN.search(
                str(result.metadata.get("section_title", ""))
            )
            section_title = str(result.metadata.get("section_title", "")).strip()
            citation = citation_by_chunk.get(result.id)
            if (
                section_person is None
                or citation is None
                or not cls._result_matches_person(result, people[0])
            ):
                continue

            lines = result.content.splitlines()
            for index, line in enumerate(lines):
                if not cls._ordered_name_tokens_match(
                    cls._normalize_person_name(people[0]).split(),
                    cls._normalize_person_name(line).split(),
                ):
                    continue

                role = next(
                    (
                        re.sub(r"[*_`]+", "", following).strip()
                        for following in lines[index + 1 :]
                        if following.strip() and not following.lstrip().startswith(("#", "!["))
                    ),
                    "",
                )
                if not role:
                    continue
                return (
                    f"The indexed JKLU page lists {section_title} as "
                    f"{role} [Source {citation.number}].",
                    (citation,),
                )

        return None

    @classmethod
    def _filter_content_to_degree(cls, content: str, target_degree: str) -> str | None:
        """Keep shared context and lines that mention only the target degree."""
        filtered_lines = []
        for line in content.splitlines():
            mentioned_degrees = cls._degree_mentions(line)
            if not mentioned_degrees or mentioned_degrees == {target_degree}:
                filtered_lines.append(line)

        scoped_content = "\n".join(filtered_lines).strip()
        if target_degree not in cls._degree_mentions(scoped_content):
            return None
        return scoped_content

    @staticmethod
    def _has_requested_fee_field(content: str, fields: set[str]) -> bool:
        """Check whether a passage contains a fee field requested by the user."""
        normalized = content.casefold()
        if fields:
            return any(re.search(rf"\b{field}\s+fees?\b", normalized) for field in fields)
        return bool(
            re.search(
                r"\b(?:academic|technology|tuition|hostel|transport|annual)\s+fees?\b"
                r"|\bfee structure\b",
                normalized,
            )
        )

    @staticmethod
    def _academic_years(text: str) -> set[str]:
        """Normalize academic-year labels such as 2026–27 and 2026/2027."""
        return {
            f"{start}-{end[-2:]}"
            for start, end in re.findall(r"\b(20\d{2})\s*[-–—/]\s*(\d{2,4})\b", text)
        }

    @classmethod
    def _build_structured_fee_answer(
        cls,
        question: str,
        results: list[RetrievalResult],
        citations: list[SourceCitation],
    ) -> tuple[str, tuple[SourceCitation, ...]] | None:
        """Render requested annual degree fees when one passage states them together."""
        if not re.search(r"\b(?:fees?|tuition|costs?|charges?)\b", question, re.I):
            return None
        if re.search(
            r"\b(?:scholarship|eligibility|compare|comparison|versus|vs\.?|between|difference)\b",
            question,
            re.I,
        ):
            return None

        degree_mentions = cls._degree_mentions(question)
        if len(degree_mentions) != 1:
            return None

        requested_fields = {
            field
            for field in FEE_AMOUNT_PATTERNS
            if re.search(rf"\b{field}\s+fees?\b", question, re.I)
        }
        if "academic" not in requested_fields and re.search(r"\btuition\s+fees?\b", question, re.I):
            requested_fields.add("academic")
        if re.search(
            r"\b(?:hostel|accommodation|transport|mess|registration|application)\b",
            question,
            re.I,
        ):
            return None
        if not requested_fields:
            requested_fields = set(FEE_AMOUNT_PATTERNS)

        target_degree = next(iter(degree_mentions))
        citation_by_chunk = {citation.chunk_id: citation for citation in citations}
        requested_years = cls._academic_years(question)
        if len(requested_years) > 1:
            return None

        for result in results:
            citation = citation_by_chunk.get(result.id)
            if citation is None or target_degree not in cls._degree_mentions(result.content):
                continue
            result_years = cls._academic_years(result.content)
            if requested_years and result_years != requested_years:
                continue
            if not requested_years and len(result_years) != 1:
                continue

            available_amounts = cls._extract_fee_amounts(result.content, target_degree)
            amounts = {field: available_amounts.get(field) for field in requested_fields}
            if not all(amounts.values()):
                continue

            year = next(iter(requested_years or result_years))
            year_label = f"For academic year {year.replace('-', '–')}, " if year else ""
            degree_label = DEGREE_LABELS[target_degree]
            if len(amounts) == 1:
                field, amount = next(iter(amounts.items()))
                field_label = "academic fee" if field == "academic" else "technology fee"
                return (
                    f"{year_label}{degree_label} annual {field_label}: "
                    f"₹{amount} [Source {citation.number}]",
                    (citation,),
                )

            lines = [f"{year_label}{degree_label} annual fees:"]
            for field in ("academic", "technology"):
                amount = amounts.get(field)
                if amount is not None:
                    label = "academic fee" if field == "academic" else "technology fee"
                    lines.append(f"- Annual {label}: ₹{amount} [Source {citation.number}]")

            return "\n".join(lines), (citation,)

        return None

    @staticmethod
    def _degree_mentions(text: str) -> set[str]:
        """Return recognized degree names mentioned in text."""
        return {degree for degree, pattern in DEGREE_PATTERNS.items() if pattern.search(text)}

    @classmethod
    def _extract_fee_amounts(cls, content: str, target_degree: str) -> dict[str, str]:
        """Extract annual fee amounts from labelled text or a degree fee table row."""
        amounts = {
            field: match.group(1)
            for field, pattern in FEE_AMOUNT_PATTERNS.items()
            if (match := pattern.search(content))
        }
        if len(amounts) == len(FEE_AMOUNT_PATTERNS):
            return amounts

        lines = content.splitlines()
        for index, line in enumerate(lines):
            headers = [cell.strip().casefold() for cell in line.split("|")]
            field_columns = {
                field: column
                for column, header in enumerate(headers)
                for field in FEE_AMOUNT_PATTERNS
                if re.search(rf"\b{field}\s+fees?\b", header)
            }
            if not field_columns:
                continue

            for row in lines[index + 1 :]:
                if "|" not in row:
                    break
                cells = [cell.strip() for cell in row.split("|")]
                if cls._degree_mentions(row) != {target_degree}:
                    continue
                for field, column in field_columns.items():
                    if column < len(cells):
                        match = re.search(r"(?:INR|₹)?\s*([\d,]+)", cells[column], re.I)
                        if match:
                            amounts[field] = match.group(1)
                break

        return amounts

    @staticmethod
    def _build_context(
        results: list[RetrievalResult],
    ) -> tuple[str, list[SourceCitation]]:
        """Format non-empty chunks and assign consistent citation numbers."""

        context_parts: list[str] = []
        citations: list[SourceCitation] = []

        for result in results:
            content = result.content.strip()

            if not content:
                continue

            title = str(result.title or "").strip()

            if title.lower() in {"", "undefined", "none", "null"}:
                title = "Untitled JKLU document"

            number = len(citations) + 1

            context_parts.append(
                f"[Source {number}]\n"
                f"Title: {title}\n"
                f"URL: {result.source_url or 'Unavailable'}\n"
                f"Chunk ID: {result.id}\n"
                f"Content:\n{content}"
            )

            citations.append(
                SourceCitation(
                    number=number,
                    title=title,
                    url=result.source_url,
                    chunk_id=result.id,
                )
            )

        return "\n\n---\n\n".join(context_parts), citations

    @staticmethod
    def _build_structured_context(
        items: list[StructuredItem],
    ) -> str:
        """Format structured programme evidence for the LLM."""

        if not items:
            return ""

        lines = [
            "VERIFIED STRUCTURED B.TECH PROGRAMME LIST:",
            "The following programme names were extracted directly "
            "from retrieved document headings.",
            "You MUST include every item exactly once.",
        ]

        for index, item in enumerate(items, start=1):
            lines.append(f"{index}. {item.name} [Chunk ID: {item.chunk_id}]")

        return "\n".join(lines)

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
        structured_context: str = "",
        history: Sequence[tuple[str, str]] = (),
    ) -> str:
        """Build a grounded prompt for JKLU Compass."""

        recent_turns = RAGPipeline._subject_history(history)
        conversation = "\n".join(f"{role}: {content[:500]}" for role, content in recent_turns)
        guidance = []

        if len(RAGPipeline._split_question_parts(question)) > 1:
            guidance.append(
                "Answer every requested part in the order asked. Use a separate short "
                "bullet for each part; if evidence is missing for one part, still answer "
                "the parts that are supported."
            )

        if BTECH_YEAR_SEMESTER_PATTERN.search(question):
            guidance.append(
                "Interpret a phrase like 'B.Tech 3 year 5th semester' as third year, "
                "Semester V (the fifth semester), not as a three-year degree."
            )
            guidance.append(
                "B.Tech semester subjects can vary by branch. If the branch is not "
                "specified, do not claim that Semester V does not exist; state the "
                "branch ambiguity and ask which branch, using only curriculum evidence "
                "that is actually present."
            )

        if CLASS_TIMING_PATTERN.search(question):
            guidance.append(
                "For daily class start or end times, do not treat academic-calendar "
                "dates (such as the day classes begin) as a daily timetable. If no daily "
                "hours are stated in the passages, say they are not available in the "
                "indexed sources."
            )
        if re.search(r"\b(?:dean|director)\b", question, re.I) and re.search(
            r"\b(?:iet|institute\s+of\s+engineering(?:\s+and|\s*&)?\s+technology)\b",
            question,
            re.I,
        ):
            guidance.append(
                "For the IET dean, prioritize the official Institute of Engineering & "
                "Technology faculty listing. Do not confuse it with deans or directors "
                "of other JKLU institutes or university-wide roles. Give the person's "
                "name and role first; add background only when the supplied passages "
                "directly support it."
            )
        question_guidance = "\n".join(f"- {item}" for item in guidance)

        return f"""You are JKLU Compass, an AI assistant for
JK Lakshmipat University (JKLU).

Answer like a helpful chat assistant: lead with the answer, use natural and concise
language, and add a short list only when it makes the answer easier to read. Do not
start with boilerplate such as "Based on the provided passages" or repeat unrelated
information from retrieved documents.

GROUNDING AND CITATIONS:

- Use the reference passages as evidence for every JKLU fact. Use the recent chat only
  to resolve references such as "he", "their", or "that"; chat is not factual evidence.
- Answer only what the current question asks. Never invent university facts or fill
  gaps with outside knowledge, or add unrelated programmes, fees, scholarships, or caveats.
- Missing evidence means "not confirmed in the indexed sources"; it does not prove
  that a programme, semester, subject, or timetable does not exist. Make a negative
  claim only when a reference passage explicitly supports it.
- Do not invent citations or URLs. Cite factual claims with exact labels, such as [Source 1].
- If the passages do not support the requested fact, say so briefly and directly.
- Distinguish degree levels and academic years. Only report a year when the evidence
  explicitly matches it.
- For a question about one degree, include only information for that degree.
- Only claim a requested academic year when the evidence explicitly supports it.
- For a named person, use only passages that identify that same person. Never transfer a
  degree, institution, date, or role from another person's profile. If the matching
  person's passages do not state a requested detail, say it is not confirmed in the
  indexed sources.
- For a requested list, include all relevant supported items. State that it may be
  incomplete only when the question asks for a complete list and the passages do not
  establish completeness.
- When verified structured evidence is supplied, include every item exactly once,
  preserving its name and citation.

QUESTION-SPECIFIC GUIDANCE:

{question_guidance or "No special interpretation is needed."}

STRUCTURED EVIDENCE:

{structured_context or "No structured programme list was extracted."}

RECENT CONVERSATION (reference resolution only; not evidence):

<conversation>
{conversation or "No earlier messages."}
</conversation>

REFERENCE PASSAGES:

<context>
{context}
</context>

USER QUESTION:

{question}

Write a direct, natural, evidence-grounded answer with citations.

ANSWER:
"""
