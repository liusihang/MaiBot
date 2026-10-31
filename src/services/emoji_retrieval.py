"""Semantic retrieval over registered emoji descriptions using the shared Pi embeddings.

Text queries and descriptions use the same observed vector space. Image-memory
vectors are deliberately not mixed into this index without a cross-modal contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import asyncio
import json
import math
import sqlite3
import time
import uuid

from src.common.paths import get_instance_root

if TYPE_CHECKING:
    from src.services.embedding_service import EmbeddingServiceClient


MAX_INDEX_ITEMS_PER_QUERY = 8
INDEX_SECONDS_PER_QUERY = 8.0
INDEX_LEASE_SECONDS = 600.0


@dataclass
class EmojiRetrievalResult:
    hits: List[Tuple[Any, float]]
    indexed: int
    total: int

    @property
    def complete(self) -> bool:
        return self.indexed == self.total


def unit_vector(values: Sequence[float]) -> List[float]:
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Emoji embedding must contain finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        raise ValueError("Emoji embedding must have nonzero norm")
    return [value / norm for value in vector]


class EmojiDescriptionIndex:
    """Persistent successful embeddings; SQLite and filesystem work stay off-loop."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        path = self.path or get_instance_root() / "data/pi_harness/emoji_description_vectors.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS vectors (space TEXT NOT NULL, content_hash TEXT NOT NULL, "
            "vector TEXT NOT NULL, PRIMARY KEY(space, content_hash))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS leases (space TEXT NOT NULL, content_hash TEXT NOT NULL, "
            "owner TEXT NOT NULL, expires REAL NOT NULL, PRIMARY KEY(space, content_hash))"
        )
        return connection

    def read(self, space: str, texts: Sequence[str]) -> Dict[str, List[float]]:
        connection = self._connect()
        try:
            result = {}
            for text in texts:
                row = connection.execute(
                    "SELECT vector FROM vectors WHERE space=? AND content_hash=?",
                    (space, sha256(text.encode()).hexdigest()),
                ).fetchone()
                if row:
                    result[text] = unit_vector(json.loads(row[0]))
            return result
        finally:
            connection.close()

    def write(self, space: str, text: str, vector: Sequence[float]) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                    (space, sha256(text.encode()).hexdigest(), json.dumps(unit_vector(vector))),
                )
        finally:
            connection.close()

    def reserve(self, space: str, text: str, owner: str) -> bool:
        """Cross-loop/process single-flight with crash recovery, no network in transaction."""
        key = sha256(text.encode()).hexdigest()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM vectors WHERE space=? AND content_hash=?", (space, key)).fetchone():
                connection.rollback()
                return False
            connection.execute("DELETE FROM leases WHERE expires < ?", (time.time(),))
            cursor = connection.execute(
                "INSERT OR IGNORE INTO leases VALUES (?, ?, ?, ?)",
                (space, key, owner, time.time() + INDEX_LEASE_SECONDS),
            )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()

    def release(self, space: str, text: str, owner: str) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "DELETE FROM leases WHERE space=? AND content_hash=? AND owner=?",
                    (space, sha256(text.encode()).hexdigest(), owner),
                )
        finally:
            connection.close()


async def _reserve(index: EmojiDescriptionIndex, space: str, text: str, owner: str) -> bool:
    # A cancelled to_thread await cannot cancel its SQLite transaction. Wait for
    # that transaction before releasing, or it could create an orphan lease later.
    task = asyncio.create_task(asyncio.to_thread(index.reserve, space, text, owner))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        await asyncio.shield(asyncio.to_thread(index.release, space, text, owner))
        raise


def _space(result: Any) -> str:
    from src.services.embedding_service import resolve_embedding_model_fingerprint

    return resolve_embedding_model_fingerprint(
        model_name=result.model_name,
        model_identifier=result.model_identifier,
        api_provider=result.api_provider,
        dimension=len(result.embedding),
    )["hash"]


def _available(emojis: Sequence[Any]) -> List[Any]:
    # The registered emoji manager is the allowlist, never the general image DB.
    unique = {}
    for emoji in emojis:
        if emoji.file_hash and emoji.description.strip() and emoji.full_path.is_file():
            unique.setdefault(emoji.file_hash, emoji)
    return list(unique.values())


async def _populate(
    texts: Sequence[str],
    vectors: Dict[str, List[float]],
    *,
    space: str,
    session_id: str,
    client: EmbeddingServiceClient,
    index: EmojiDescriptionIndex,
    max_items: int,
    max_seconds: float,
) -> Dict[str, List[float]]:
    # Every successful vector is persisted, so bounded warmup can safely resume.
    # Concurrent loops claim different missing entries rather than paying twice.
    owner = uuid.uuid4().hex
    deadline = time.monotonic() + max_seconds
    attempted = 0
    for text in texts:
        if text in vectors:
            continue
        remaining = deadline - time.monotonic()
        if attempted >= max_items or remaining <= 0:
            break
        acquired = await _reserve(index, space, text, owner)
        if not acquired:
            continue
        attempted += 1
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            result = await asyncio.wait_for(
                client.embed_text(text, session_id=session_id),
                timeout=min(remaining, INDEX_SECONDS_PER_QUERY),
            )
            if _space(result) != space:
                raise ValueError("Emoji query and description embeddings use different vector spaces")
            vector = unit_vector(result.embedding)
            await asyncio.to_thread(index.write, space, text, vector)
        except TimeoutError:
            break
        finally:
            # Cancellation must not leave the lease blocking the next request.
            await asyncio.shield(asyncio.to_thread(index.release, space, text, owner))
    return await asyncio.to_thread(index.read, space, texts)


def _rank(
    candidates: Sequence[Any],
    vectors: Dict[str, List[float]],
    query_vector: Sequence[float],
) -> List[Tuple[Any, float]]:
    ranked = []
    for emoji in candidates:
        vector = vectors.get(emoji.description.strip())
        if vector is None:
            continue
        if len(vector) != len(query_vector):
            raise ValueError("Emoji cached vector dimension does not match query")
        score = sum(a * b for a, b in zip(query_vector, vector, strict=True))
        if score > 0:
            ranked.append((emoji, score))
    # Stable ties avoid random picks. The Agent may abstain even on the top hit.
    ranked.sort(key=lambda item: (-item[1], item[0].file_hash))
    return ranked[:3]


async def query_registered_emojis(
    query: str,
    emojis: Sequence[Any],
    *,
    session_id: str,
    client: Optional[EmbeddingServiceClient] = None,
    index: Optional[EmojiDescriptionIndex] = None,
) -> EmojiRetrievalResult:
    """Return at most three positive-cosine registered resources, without an LLM.

    Missing vectors are populated once and reused across restarts. A provider or
    model change creates a new namespace; mismatched observed routes fail loudly.
    """
    query = query.strip()
    if not query or len(query) > 1000:
        raise ValueError("Emoji query must contain 1–1000 characters")
    candidates = await asyncio.to_thread(_available, emojis)
    if not candidates:
        return EmojiRetrievalResult([], 0, 0)
    if client is None:
        from src.services.embedding_service import EmbeddingServiceClient

        client = EmbeddingServiceClient(task_name="embedding", request_type="emoji.query")
    index = index or EmojiDescriptionIndex()
    query_result = await client.embed_text(query, session_id=session_id)
    query_vector = unit_vector(query_result.embedding)
    space = _space(query_result)
    texts = list(dict.fromkeys(emoji.description.strip() for emoji in candidates))
    vectors = await asyncio.to_thread(index.read, space, texts)
    vectors = await _populate(
        texts,
        vectors,
        space=space,
        session_id=session_id,
        client=client,
        index=index,
        max_items=MAX_INDEX_ITEMS_PER_QUERY,
        max_seconds=INDEX_SECONDS_PER_QUERY,
    )
    ranked = await asyncio.to_thread(_rank, candidates, vectors, query_vector)
    return EmojiRetrievalResult(ranked, len(vectors), len(texts))


async def warm_registered_emojis(
    emojis: Sequence[Any],
    *,
    session_id: str,
    client: Optional[EmbeddingServiceClient] = None,
    index: Optional[EmojiDescriptionIndex] = None,
    max_items: int = 1000,
    max_seconds: float = 300.0,
) -> EmojiRetrievalResult:
    """Explicit operator warmup, not a chat query or automatic paid background job.

    Returns progress without candidates. Repeat when incomplete; saved vectors and
    shared leases are reused. Limits bound each invocation, including a route probe.
    """
    if max_items < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Warmup requires positive item and time budgets")
    candidates = await asyncio.to_thread(_available, emojis)
    texts = list(dict.fromkeys(emoji.description.strip() for emoji in candidates))
    if not texts:
        return EmojiRetrievalResult([], 0, 0)
    if client is None:
        from src.services.embedding_service import EmbeddingServiceClient

        client = EmbeddingServiceClient(task_name="embedding", request_type="emoji.warmup")
    index = index or EmojiDescriptionIndex()
    # One probe identifies the actual provider/model space, never a guessed route.
    started = time.monotonic()
    probe = await asyncio.wait_for(
        client.embed_text("emoji index", session_id=session_id), timeout=min(max_seconds, INDEX_SECONDS_PER_QUERY)
    )
    unit_vector(probe.embedding)
    space = _space(probe)
    vectors = await asyncio.to_thread(index.read, space, texts)
    vectors = await _populate(
        texts,
        vectors,
        space=space,
        session_id=session_id,
        client=client,
        index=index,
        max_items=max_items,
        max_seconds=max(0.0, max_seconds - (time.monotonic() - started)),
    )
    return EmojiRetrievalResult([], len(vectors), len(texts))
