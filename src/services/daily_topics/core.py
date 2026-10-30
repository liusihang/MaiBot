"""Offline-testable nightly topic fold. The only model boundary is ``summarize``.

A model receives a full current board plus one lossless source batch, and must
return a replacement board (including [] to clear it). No partial board is ever
published. Database work is offloaded from the event loop.
"""

from datetime import date
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
import asyncio
import json
import math
import time

from .store import TopicStore, canonical_json


Topic = Dict[str, Any]
Source = Dict[str, Any]
Summarizer = Callable[[List[Topic], List[Source], Dict[str, Any]], Awaitable[List[Topic]]]


def source_batches(records: List[Source], max_chars: int) -> List[List[Source]]:
    """Split all text losslessly; offsets are Unicode character positions.

    ``max_chars`` bounds source text per batch (not the JSON metadata envelope).
    Empty text remains a source part and therefore receives explicit coverage.
    Splits and ordering depend only on the frozen sources and frozen batch size.
    """
    batches: List[List[Source]] = []
    batch: List[Source] = []
    used = 0
    for record in records:
        text = record["text"]
        count = max(1, (len(text) + max_chars - 1) // max_chars)
        for index in range(count):
            offset = index * max_chars
            piece = text[offset : offset + max_chars]
            if batch and (used + len(piece) > max_chars or len(batch) >= 100):
                batches.append(batch)
                batch, used = [], 0
            batch.append(
                {
                    **record,
                    "text": piece,
                    "offset": offset,
                    "end_offset": offset + len(piece),
                    "total_chars": len(text),
                    "part_index": index,
                    "part_count": count,
                }
            )
            used += len(piece)
    if batch:
        batches.append(batch)
    return batches


def validate_topics(value: Any, allowed_evidence: Set[str], max_chars: int) -> List[Topic]:
    """Validate structure and provenance; factual accuracy still needs review.

    Source text is untrusted data. An invented/cross-session evidence ID is a
    hard failure, never silently stripped or accepted as model instructions.
    """
    if not isinstance(value, list):
        raise ValueError("summarizer must return a JSON topic list")
    normalized = []
    seen = set()
    fields = {"id", "topic", "background", "evidence_ids"}
    for item in value:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("each topic requires exactly id, topic, background, evidence_ids")
        topic = {}
        for field in ("id", "topic", "background"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"topic {field} must be a nonempty string")
            topic[field] = item[field].strip()
        if len(topic["id"]) > 128 or topic["id"] in seen:
            raise ValueError("topic IDs must be unique and at most 128 characters")
        seen.add(topic["id"])
        evidence = item["evidence_ids"]
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(source_id, str) for source_id in evidence)
        ):
            raise ValueError("a topic requires nonempty string evidence_ids")
        if any(source_id not in allowed_evidence for source_id in evidence):
            raise ValueError("topic cites unknown, unseen, or cross-session evidence")
        topic["evidence_ids"] = sorted(set(evidence))
        normalized.append(topic)
    normalized.sort(key=lambda topic: topic["id"])
    if len(canonical_json(normalized)) > max_chars:
        raise ValueError("topic board exceeds max_topics_chars; no truncation is permitted")
    return normalized


class DailyTopicProcessor:
    def __init__(
        self,
        store: TopicStore,
        summarize: Summarizer,
        max_batch_chars: int = 16000,
        max_topics_chars: int = 6000,
        max_batches_per_pass: int = 8,
        lease_seconds: float = 300,
        publication_cycle: Optional[Callable[[float], str]] = None,
    ) -> None:
        for name, value in (
            ("max_batch_chars", max_batch_chars),
            ("max_topics_chars", max_topics_chars),
            ("max_batches_per_pass", max_batches_per_pass),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive and finite")
        self.store = store
        self.summarize = summarize
        self.max_batch_chars = max_batch_chars
        self.max_topics_chars = max_topics_chars
        self.max_batches_per_pass = max_batches_per_pass
        self.lease_seconds = lease_seconds
        self.publication_cycle = publication_cycle

    def _previous_evidence(self, session_id: str, topics: List[Topic]) -> List[Dict[str, Any]]:
        # Publication date is not claim time: late source arrivals may predate
        # newer corrections already retained in the board. Never infer event time.
        source_ids = sorted({source_id for topic in topics for source_id in topic["evidence_ids"]})
        metadata = []
        for offset in range(0, len(source_ids), 100):
            metadata.extend(self.store.read_evidence_metadata(session_id, source_ids[offset : offset + 100]))
        if {row["source_id"] for row in metadata} != set(source_ids):
            raise ValueError("previous topic references missing source evidence")
        return sorted(metadata, key=lambda row: row["source_id"])

    async def _summarize(
        self, topics: List[Topic], parts: List[Source], info: Dict[str, Any], token: str, now: Callable[[], float]
    ) -> List[Topic]:
        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(self.lease_seconds / 3)
                await asyncio.to_thread(self.store.renew, info["session_id"], token, now(), self.lease_seconds)

        model = asyncio.create_task(self.summarize(topics, parts, info))
        renewal = asyncio.create_task(heartbeat())
        try:
            done, _ = await asyncio.wait((model, renewal), return_when=asyncio.FIRST_COMPLETED)
            if renewal in done:
                await renewal  # Lease loss must interrupt, never publish a stale result.
            return await model
        finally:
            for task in (model, renewal):
                if not task.done():
                    task.cancel()
            await asyncio.gather(model, renewal, return_exceptions=True)

    async def process(self, session_id: str, cycle_day: str, cutoff: float, now: float) -> Dict[str, Any]:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must identify an existing real chat session")
        if date.fromisoformat(cycle_day).isoformat() != cycle_day:
            raise ValueError("cycle_day must be YYYY-MM-DD")
        if not math.isfinite(cutoff) or not math.isfinite(now):
            raise ValueError("cutoff and now must be finite epoch timestamps")
        started = time.monotonic()

        def clock() -> float:
            return now + time.monotonic() - started

        acquired = await asyncio.to_thread(
            self.store.acquire, session_id, cycle_day, cutoff, clock(), self.lease_seconds, self.max_batch_chars
        )
        if acquired["status"] != "working":
            job = acquired.get("job", {})
            return {
                "status": acquired["status"],
                "session_id": session_id,
                "cycle_day": acquired["cycle_day"],
                "published_cycle_day": job.get("released_cycle_day"),
                "batches_processed": 0,
                "version": job.get("published_version"),
                "changed": False,
            }
        token = acquired["token"]
        job = acquired["job"]
        completed = 0
        try:
            batches = await asyncio.to_thread(source_batches, acquired["sources"], job["batch_chars"])
            # Even an idle board gets one daily opportunity to discard stale topics.
            # Freeze this decision from the durable job: after [] is staged, a retry
            # must still consider that single empty-source batch already processed.
            if not batches and (job["topics"] or job["next_batch"]):
                batches = [[]]
            topics, revision = job["topics"], job["revision"]
            next_batch = job["next_batch"]
            while next_batch < len(batches) and completed < self.max_batches_per_pass:
                parts = batches[next_batch]
                allowed = {source_id for topic in topics for source_id in topic["evidence_ids"]}
                allowed.update(part["source_id"] for part in parts)
                info = {
                    "session_id": session_id,
                    "cycle_day": job["cycle_day"],
                    "cutoff": job["cutoff"],
                    "now": now,
                    "batch_index": next_batch,
                    "total_batches": len(batches),
                    "previous_as_of_day": acquired["previous_as_of_day"],
                    "previous_evidence": await asyncio.to_thread(self._previous_evidence, session_id, topics),
                    "published_cycle_day": cycle_day,
                }
                # Pass copies: a buggy adapter must not mutate accepted stage/evidence.
                proposed = await self._summarize(
                    json.loads(canonical_json(topics)), json.loads(canonical_json(parts)), info, token, clock
                )
                topics = validate_topics(proposed, allowed, self.max_topics_chars)
                await asyncio.to_thread(
                    self.store.stage, session_id, job["cycle_day"], token, revision, topics, clock()
                )
                revision += 1
                next_batch += 1
                completed += 1
            if next_batch < len(batches):
                current = await asyncio.to_thread(self.store.current, session_id)
                return {
                    "status": "incomplete",
                    "session_id": session_id,
                    "cycle_day": job["cycle_day"],
                    "published_cycle_day": None,
                    "batches_processed": completed,
                    "remaining_batches": len(batches) - next_batch,
                    "version": current["version"] if current else None,
                    "changed": False,
                }
            finished_at = clock()
            released_cycle_day = cycle_day
            if self.publication_cycle is not None:
                actual_cycle = self.publication_cycle(finished_at)
                if date.fromisoformat(actual_cycle).isoformat() != actual_cycle:
                    raise ValueError("publication_cycle must return YYYY-MM-DD")
                # A slow old job crossing the local publish boundary consumes
                # the new day's single release slot, not a second stale slot.
                released_cycle_day = max(cycle_day, actual_cycle)
            result = await asyncio.to_thread(
                self.store.finish,
                session_id,
                job["cycle_day"],
                token,
                revision,
                len(batches),
                finished_at,
                released_cycle_day,
            )
            return {**result, "session_id": session_id, "batches_processed": completed}
        finally:
            await asyncio.to_thread(self.store.release, session_id, token)
