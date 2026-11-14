"""Retrieval-augmented generation pipeline for JKLU Compass."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

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
        results_by_query = [
            self.retriever.retrieve(retrieval_query, top_k=self.top_k)
            for retrieval_query in retrieval_queries
        ]
        results = self._merge_retrieval_results(results_by_query, self.top_k)
        context_results = (
            results
            if len(retrieval_queries) > 1
            else self._scope_results_to_question(question, results)
        )

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
        parts = [part.strip(" ,;?") for part in QUESTION_PART_SEPARATOR.split(question)]
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
        question_guidance = "\n".join(f"- {item}" for item in guidance)

        return f"""You are JKLU Compass, an AI assistant for
JK Lakshmipat University (JKLU).

Answer like a helpful chat assistant: lead with the answer, use natural and concise
language, and add a short list only when it makes the answer easier to read. Do not
start with boilerplate such as "Based on the provided passages" or repeat unrelated
information from retrieved documents.

GROUNDING AND CITATIONS:

- Use the recent chat only to resolve references such as "he", "their", or "that".
  The chat is not factual evidence and must not be cited. Use the reference passages
  below for every JKLU fact.
- Answer only what the current question asks. Never invent facts, fill gaps with
  outside knowledge, or add unrelated programmes, fees, scholarships, or caveats.
- Missing evidence means "not confirmed in the indexed sources"; it does not prove
  that a programme, semester, subject, or timetable does not exist. Make a negative
  claim only when a reference passage explicitly supports it.
- Cite each factual claim with the exact supplied label, such as [Source 1]. Do not
  invent source labels or URLs.
- If the passages do not support the requested fact, say so briefly and directly.
- Distinguish degree levels and academic years. Only report a year when the evidence
  explicitly matches it.
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
