"""Extract structured university entities from retrieved document metadata."""

from __future__ import annotations

from dataclasses import dataclass

from jklu_compass.retrieval.retriever import RetrievalResult


@dataclass(frozen=True, slots=True)
class StructuredItem:
    """A structured item extracted from a retrieved document."""

    name: str
    chunk_id: str
    source_url: str | None
    source_title: str


class StructuredExtractor:
    """Extract structured entities without modifying source documents."""

    @staticmethod
    def extract_btech_programmes(
        results: list[RetrievalResult],
    ) -> list[StructuredItem]:
        """Extract B.Tech programme headings in document order."""

        items: list[StructuredItem] = []
        seen: set[str] = set()

        for result in results:
            metadata = result.metadata or {}

            heading_path = metadata.get("heading_path", [])
            section_title = str(metadata.get("section_title", "")).strip()
            heading_level = metadata.get("heading_level")

            if isinstance(heading_path, str):
                # Some older records may store the heading path as text.
                try:
                    import ast

                    parsed = ast.literal_eval(heading_path)
                    if isinstance(parsed, list):
                        heading_path = parsed
                except (ValueError, SyntaxError):
                    heading_path = heading_path.split(" > ")

            if not isinstance(heading_path, list):
                heading_path = []

            normalized_path = [str(heading).strip().lstrip("#").strip() for heading in heading_path]

            has_btech_parent = any(
                "B.TECH" in heading.upper() or "BACHELOR OF TECHNOLOGY" in heading.upper()
                for heading in normalized_path
            )

            if not has_btech_parent:
                continue

            if heading_level is not None and int(heading_level) != 5:
                continue

            if not section_title:
                continue

            # Remove duplicate programme names while preserving order.
            normalized_name = " ".join(section_title.casefold().split())

            if normalized_name in seen:
                continue

            seen.add(normalized_name)

            items.append(
                StructuredItem(
                    name=section_title,
                    chunk_id=result.id,
                    source_url=result.source_url,
                    source_title=result.title,
                )
            )

        return items
