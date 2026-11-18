"""Reconstruct a clarified question as a standalone RAG search query."""

from __future__ import annotations

from collections.abc import Sequence


class QueryReconstructor:
    """Combine the original request, selected intent, and relevant chat context."""

    @staticmethod
    def reconstruct(
        original_question: str,
        selected_option: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> str:
        query = (
            f"{original_question.strip()} {selected_option.strip()}"
        )
        relevant_history = [
            (role, content.strip()[:500])
            for role, content in history[-8:]
            if role in ("user", "assistant") and content.strip()
        ]
        if relevant_history:
            conversation = "\n".join(f"{role}: {content}" for role, content in relevant_history)
            query += f"\nRelevant conversation for context resolution:\n{conversation}"
        return query
