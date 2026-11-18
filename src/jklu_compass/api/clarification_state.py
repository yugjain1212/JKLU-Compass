"""Short-lived, single-use pending clarification state."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Literal

from jklu_compass.rag.query_clarifier import ClarificationOption

PendingStatus = Literal["ready", "used", "stale", "expired"]


@dataclass(slots=True)
class PendingClarification:
    conversation_id: str
    clarification_id: str
    question: str
    history: tuple[tuple[str, str], ...]
    options: tuple[ClarificationOption, ...]
    expires_at: float
    status: PendingStatus = "ready"


class ClarificationState:
    """Keep only one current clarification per conversation and reject replays."""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = RLock()
        self._items: dict[str, PendingClarification] = {}

    def create(
        self,
        conversation_id: str,
        clarification_id: str,
        question: str,
        history: tuple[tuple[str, str], ...],
        options: tuple[ClarificationOption, ...],
    ) -> PendingClarification:
        with self._lock:
            self._prune()
            previous = self._items.get(conversation_id)
            if previous is not None and previous.status == "ready":
                previous.status = "stale"
            pending = PendingClarification(
                conversation_id=conversation_id,
                clarification_id=clarification_id,
                question=question,
                history=history,
                options=options,
                expires_at=monotonic() + self.ttl_seconds,
            )
            self._items[conversation_id] = pending
            return pending

    def consume(
        self,
        conversation_id: str,
        clarification_id: str,
        option_id: str,
        custom_answer: str | None = None,
    ) -> tuple[PendingClarification, ClarificationOption]:
        with self._lock:
            pending = self._items.get(conversation_id)
            if pending is None or pending.clarification_id != clarification_id:
                raise LookupError("stale")
            if monotonic() >= pending.expires_at:
                pending.status = "expired"
                raise TimeoutError("expired")
            if pending.status == "used":
                raise RuntimeError("used")
            if pending.status != "ready":
                raise LookupError("stale")
            if option_id == "other":
                if not custom_answer or not custom_answer.strip() or len(custom_answer) > 500:
                    raise ValueError("invalid_custom_answer")
                option = ClarificationOption(id="other", question=custom_answer.strip())
            elif custom_answer is not None:
                raise ValueError("unexpected_custom_answer")
            else:
                option = next((item for item in pending.options if item.id == option_id), None)
            if option is None:
                raise ValueError("invalid_option")
            pending.status = "used"
            return pending, option

    def _prune(self) -> None:
        now = monotonic()
        for item in self._items.values():
            if item.status == "ready" and item.expires_at <= now:
                item.status = "expired"
        self._items = {
            key: item
            for key, item in self._items.items()
            if item.expires_at + self.ttl_seconds > now
        }
