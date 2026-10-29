"""Durable, host-scoped canonical threads, reference deltas, and execution receipts.

All methods are synchronous SQLite operations. Async callers MUST use
``asyncio.to_thread`` (including construction). A stream id is the host's existing
ChatSession.session_id, never a group selected by a model/tool argument. Nothing
here plans tasks, rewrites old messages, or executes a tool.

The immutable event archive is separate from the visible head. Compaction swaps
only an exact, tool-safe prefix; concurrent appends survive. A failed/cancelled
summary needs no rollback because begin_compaction does not change the head.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import hashlib
import json
import math
import sqlite3
import uuid


class LedgerConflict(RuntimeError):
    """An expected revision, immutable source event, or compaction prefix changed."""


class UnsafeCompaction(LedgerConflict):
    """A prefix would split a tool call/result pair or hide an unresolved execution."""


@dataclass(frozen=True)
class EventInput:
    event_id: str
    payload: Any
    kind: str = "native"
    tool_call_ids: Tuple[str, ...] = ()
    tool_result_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    kind: str
    payload_json: str
    reference_key: Optional[str] = None
    reference_source_key: Optional[str] = None
    represented_reference_keys: Tuple[str, ...] = ()
    tool_call_ids: Tuple[str, ...] = ()
    tool_result_ids: Tuple[str, ...] = ()

    @property
    def payload(self) -> Any:
        # Returning a fresh value prevents a caller mutating a snapshot/plan.
        return json.loads(self.payload_json)


@dataclass(frozen=True)
class ThreadSnapshot:
    stream_id: str
    revision: int
    events: Tuple[LedgerEvent, ...]


@dataclass(frozen=True)
class ReferenceAppendResult:
    snapshot: ThreadSnapshot
    appended: bool
    event: Optional[LedgerEvent]
    reference_key: str


@dataclass(frozen=True)
class CompactionPlan:
    token: str
    stream_id: str
    base_revision: int
    prefix: Tuple[LedgerEvent, ...]
    leading: Tuple[LedgerEvent, ...] = ()


@dataclass(frozen=True)
class ToolExecution:
    stream_id: str
    call_id: str
    status: str
    intent_json: str
    result_event_id: Optional[str] = None
    result_json: Optional[str] = None

    @property
    def intent(self) -> Any:
        return json.loads(self.intent_json)

    @property
    def result(self) -> Any:
        """Known terminal receipt, or partial observation when status is unknown."""
        return None if self.result_json is None else json.loads(self.result_json)


def _require_id(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty host/source identifier")


def _json_text(value: Any, *, canonical: bool = False) -> str:
    def validate(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                validate(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                validate(child)
            return
        raise TypeError("ledger payloads must be native JSON values (no lossy conversion)")

    validate(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=canonical)


def _native_tool_ids(payload: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Recognize Pi native messages, not arbitrary nested/reference content."""
    if not isinstance(payload, dict):
        return (), ()
    calls: List[str] = []
    results: List[str] = []
    if payload.get("role") == "assistant":
        content = payload.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    call_id = block.get("id")
                    if isinstance(call_id, str) and call_id:
                        calls.append(call_id)
    if payload.get("role") == "toolResult":
        call_id = payload.get("toolCallId")
        if isinstance(call_id, str) and call_id:
            results.append(call_id)
    return tuple(calls), tuple(results)


def _input_event(value: EventInput) -> LedgerEvent:
    _require_id(value.event_id, "event_id")
    _require_id(value.kind, "kind")
    payload_json = _json_text(value.payload)
    inferred_calls, inferred_results = _native_tool_ids(value.payload) if value.kind == "native" else ((), ())
    calls = tuple(dict.fromkeys((*value.tool_call_ids, *inferred_calls)))
    results = tuple(dict.fromkeys((*value.tool_result_ids, *inferred_results)))
    for call_id in (*calls, *results):
        _require_id(call_id, "tool call id")
    return LedgerEvent(value.event_id, value.kind, payload_json, tool_call_ids=calls, tool_result_ids=results)


class ThreadLedger:
    """File-backed SQLite ledger safe to share across threads and processes.

    Reusing an event id with different bytes/metadata is a conflict, not an edit.
    Replaying an archived event never makes it visible again. A source reference
    may, however, be reinjected after compaction if no visible event represents it.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        if str(db_path) == ":memory:":
            raise ValueError("ThreadLedger requires a durable file, not :memory:")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS thread_heads (
                    stream_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                    visible_ids_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS thread_events (
                    ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL REFERENCES thread_heads(stream_id),
                    event_id TEXT NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
                    reference_key TEXT, reference_source_key TEXT, metadata_json TEXT NOT NULL,
                    UNIQUE(stream_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS thread_reference_keys
                    ON thread_events(stream_id, reference_key);
                CREATE INDEX IF NOT EXISTS thread_reference_sources
                    ON thread_events(stream_id, reference_source_key);
                CREATE TRIGGER IF NOT EXISTS immutable_thread_events_update
                    BEFORE UPDATE ON thread_events BEGIN SELECT RAISE(ABORT, 'immutable thread event'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_thread_events_delete
                    BEFORE DELETE ON thread_events BEGIN SELECT RAISE(ABORT, 'immutable thread event'); END;
                CREATE TABLE IF NOT EXISTS thread_inbox (
                    ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL REFERENCES thread_heads(stream_id),
                    event_id TEXT NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
                    reference_key TEXT, reference_source_key TEXT, metadata_json TEXT NOT NULL,
                    UNIQUE(stream_id, event_id)
                );
                CREATE TRIGGER IF NOT EXISTS immutable_thread_inbox_update
                    BEFORE UPDATE ON thread_inbox BEGIN SELECT RAISE(ABORT, 'immutable pending input'); END;
                CREATE TABLE IF NOT EXISTS thread_compactions (
                    token TEXT PRIMARY KEY, stream_id TEXT NOT NULL,
                    plan_json TEXT NOT NULL, summary_json TEXT NOT NULL,
                    retained_keys_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_executions (
                    stream_id TEXT NOT NULL REFERENCES thread_heads(stream_id),
                    call_id TEXT NOT NULL, intent_json TEXT NOT NULL, status TEXT NOT NULL,
                    result_event_id TEXT, result_json TEXT,
                    PRIMARY KEY(stream_id, call_id),
                    CHECK(status IN ('prepared', 'executing', 'unknown', 'succeeded', 'failed'))
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _state(connection: sqlite3.Connection, stream_id: str, *, create: bool = False) -> Tuple[int, List[str]]:
        _require_id(stream_id, "stream_id")
        if create:
            connection.execute("INSERT OR IGNORE INTO thread_heads VALUES (?, 0, '[]')", (stream_id,))
        row = connection.execute("SELECT * FROM thread_heads WHERE stream_id = ?", (stream_id,)).fetchone()
        return (0, []) if row is None else (row["revision"], json.loads(row["visible_ids_json"]))

    @staticmethod
    def _from_row(row: sqlite3.Row) -> LedgerEvent:
        metadata = json.loads(row["metadata_json"])
        return LedgerEvent(
            row["event_id"],
            row["kind"],
            row["payload_json"],
            row["reference_key"],
            row["reference_source_key"],
            tuple(metadata["represented_reference_keys"]),
            tuple(metadata["tool_call_ids"]),
            tuple(metadata["tool_result_ids"]),
        )

    def _events(self, connection: sqlite3.Connection, stream_id: str, ids: Sequence[str]) -> Tuple[LedgerEvent, ...]:
        found: Dict[str, LedgerEvent] = {}
        for offset in range(0, len(ids), 400):
            chunk = ids[offset : offset + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT * FROM thread_events WHERE stream_id = ? AND event_id IN ({placeholders})",  # noqa: S608
                (stream_id, *chunk),
            )
            found.update((row["event_id"], self._from_row(row)) for row in rows)
        missing = set(ids).difference(found)
        if missing:
            raise KeyError(f"events not found in stream {stream_id!r}: {sorted(missing)!r}")
        return tuple(found[event_id] for event_id in ids)

    def _snapshot(self, connection: sqlite3.Connection, stream_id: str) -> ThreadSnapshot:
        revision, ids = self._state(connection, stream_id)
        return ThreadSnapshot(stream_id, revision, self._events(connection, stream_id, ids))

    @staticmethod
    def _check_revision(actual: int, expected: Optional[int]) -> None:
        if expected is not None and actual != expected:
            raise LedgerConflict(f"expected thread revision {expected}, found {actual}")

    @staticmethod
    def _replace_head(connection: sqlite3.Connection, stream_id: str, revision: int, ids: List[str]) -> None:
        cursor = connection.execute(
            "UPDATE thread_heads SET revision = revision + 1, visible_ids_json = ? "
            "WHERE stream_id = ? AND revision = ?",
            (_json_text(ids), stream_id, revision),
        )
        if cursor.rowcount != 1:
            raise LedgerConflict("thread head compare-and-swap failed")

    def _insert_event(self, connection: sqlite3.Connection, stream_id: str, event: LedgerEvent) -> bool:
        row = connection.execute(
            "SELECT * FROM thread_events WHERE stream_id = ? AND event_id = ?",
            (stream_id, event.event_id),
        ).fetchone()
        if row is not None:
            if self._from_row(row) != event:
                raise LedgerConflict(f"immutable event id reused with changed content: {event.event_id}")
            return False
        metadata = {
            "represented_reference_keys": list(event.represented_reference_keys),
            "tool_call_ids": list(event.tool_call_ids),
            "tool_result_ids": list(event.tool_result_ids),
        }
        connection.execute(
            "INSERT INTO thread_events (stream_id, event_id, kind, payload_json, reference_key, "
            "reference_source_key, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                stream_id,
                event.event_id,
                event.kind,
                event.payload_json,
                event.reference_key,
                event.reference_source_key,
                _json_text(metadata),
            ),
        )
        return True

    def snapshot(self, stream_id: str) -> ThreadSnapshot:
        with self._transaction() as connection:
            return self._snapshot(connection, stream_id)

    def append_event(
        self,
        stream_id: str,
        event_id: str,
        payload: Any,
        *,
        kind: str = "native",
        expected_revision: Optional[int] = None,
        tool_call_ids: Tuple[str, ...] = (),
        tool_result_ids: Tuple[str, ...] = (),
    ) -> ThreadSnapshot:
        return self.append_events(
            stream_id,
            [EventInput(event_id, payload, kind, tool_call_ids, tool_result_ids)],
            expected_revision=expected_revision,
        )

    def append_events(
        self,
        stream_id: str,
        events: Iterable[EventInput],
        *,
        expected_revision: Optional[int] = None,
    ) -> ThreadSnapshot:
        values = tuple(_input_event(value) for value in events)
        with self._transaction(write=True) as connection:
            revision, visible = self._state(connection, stream_id, create=True)
            self._check_revision(revision, expected_revision)
            changed = False
            for event in values:
                if self._insert_event(connection, stream_id, event):
                    visible.append(event.event_id)
                    changed = True
            if changed:
                self._replace_head(connection, stream_id, revision, visible)
            return self._snapshot(connection, stream_id)

    def archive_events(self, stream_id: str, event_ids: Iterable[str]) -> Tuple[LedgerEvent, ...]:
        """Read exact originals (including compacted ones) within one host stream."""
        _require_id(stream_id, "stream_id")
        with self._transaction() as connection:
            return self._events(connection, stream_id, tuple(event_ids))

    def has_event(self, stream_id: str, event_id: str) -> bool:
        """Permanent source-delivery dedup, unlike visible-reference dedup."""
        _require_id(stream_id, "stream_id")
        _require_id(event_id, "event_id")
        with self._transaction() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM thread_events WHERE stream_id = ? AND event_id = ?",
                    (stream_id, event_id),
                ).fetchone()
                is not None
            )

    def enqueue_input(self, stream_id: str, event_id: str, payload: Any, *, kind: str = "native") -> bool:
        """Freeze source ingress before debounce/Luna, without starting a model.

        Returns True only for a new durable inbox item. Pending input is not yet
        part of the visible model transcript. Neither duplicate delivery nor a
        restart can rewrite its frozen payload or resurrect an archived source.
        """
        event = _input_event(EventInput(event_id, payload, kind))
        with self._transaction(write=True) as connection:
            self._state(connection, stream_id, create=True)
            for table in ("thread_events", "thread_inbox"):
                row = connection.execute(
                    f"SELECT * FROM {table} WHERE stream_id = ? AND event_id = ?",
                    (stream_id, event_id),
                ).fetchone()
                if row is not None:
                    if self._from_row(row) != event:
                        raise LedgerConflict(f"immutable source input reused with changed content: {event_id}")
                    return False
            metadata = {
                "represented_reference_keys": [],
                "tool_call_ids": list(event.tool_call_ids),
                "tool_result_ids": list(event.tool_result_ids),
            }
            connection.execute(
                "INSERT INTO thread_inbox (stream_id, event_id, kind, payload_json, reference_key, "
                "reference_source_key, metadata_json) VALUES (?, ?, ?, ?, NULL, NULL, ?)",
                (stream_id, event_id, event.kind, event.payload_json, _json_text(metadata)),
            )
            return True

    def pending_inputs(self, stream_id: str) -> Tuple[LedgerEvent, ...]:
        """Return exact durable ingress in its original arrival order."""
        _require_id(stream_id, "stream_id")
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM thread_inbox WHERE stream_id = ? ORDER BY ordinal",
                (stream_id,),
            )
            return tuple(self._from_row(row) for row in rows)

    def consume_input(
        self,
        stream_id: str,
        event_id: str,
        *,
        expected_revision: Optional[int] = None,
    ) -> ThreadSnapshot:
        """Atomically move the oldest pending input to a tool-safe canonical tail.

        The caller cannot silently consume/reorder a newer source first. Failures
        leave the inbox unchanged. A replay of an already consumed id is harmless,
        including when its original event was subsequently compacted.
        """
        _require_id(event_id, "event_id")
        with self._transaction(write=True) as connection:
            revision, visible = self._state(connection, stream_id)
            self._check_revision(revision, expected_revision)
            row = connection.execute(
                "SELECT * FROM thread_inbox WHERE stream_id = ? AND event_id = ?",
                (stream_id, event_id),
            ).fetchone()
            if row is None:
                if (
                    connection.execute(
                        "SELECT 1 FROM thread_events WHERE stream_id = ? AND event_id = ?",
                        (stream_id, event_id),
                    ).fetchone()
                    is None
                ):
                    raise KeyError(f"pending source input not found in stream {stream_id!r}: {event_id!r}")
                return self._snapshot(connection, stream_id)
            oldest = connection.execute(
                "SELECT event_id FROM thread_inbox WHERE stream_id = ? ORDER BY ordinal LIMIT 1",
                (stream_id,),
            ).fetchone()
            if oldest["event_id"] != event_id:
                raise LedgerConflict("pending source inputs must be consumed in original arrival order")
            open_calls = set()
            for existing in self._events(connection, stream_id, visible):
                open_calls.update(existing.tool_call_ids)
                for call_id in existing.tool_result_ids:
                    if call_id not in open_calls:
                        raise LedgerConflict("source intake requires a closed, valid tool call/result boundary")
                    open_calls.remove(call_id)
            if (
                open_calls
                or connection.execute(
                    "SELECT 1 FROM tool_executions WHERE stream_id = ? AND status IN ('executing', 'unknown') LIMIT 1",
                    (stream_id,),
                ).fetchone()
                is not None
            ):
                raise LedgerConflict("source intake requires a closed tool boundary without uncertain executions")
            event = self._from_row(row)
            if self._insert_event(connection, stream_id, event):
                self._replace_head(connection, stream_id, revision, [*visible, event_id])
            connection.execute("DELETE FROM thread_inbox WHERE stream_id = ? AND event_id = ?", (stream_id, event_id))
            return self._snapshot(connection, stream_id)

    def append_reference(
        self,
        stream_id: str,
        namespace: str,
        source_id: Optional[str],
        version: Optional[str],
        content: Any,
        *,
        fallback_scope: Optional[str] = None,
        force: bool = False,
        expected_revision: Optional[int] = None,
    ) -> ReferenceAppendResult:
        """Append an automatic delta; explicit reads use force=True to bypass dedup.

        A fingerprint is explicitly NOT a stable source id. Missing either the
        source id or version requires an explicit caller-provided fallback scope,
        and hashes the complete content, never a truncated excerpt.
        """
        _require_id(namespace, "reference namespace")
        content_json = _json_text(content, canonical=True)
        if source_id and version:
            _require_id(source_id, "source_id")
            _require_id(version, "version")
            key = _json_text(["source", namespace, source_id, version])
            source_key = _json_text(["source", namespace, source_id])
            identity = {"kind": "source", "stable": True}
        else:
            if fallback_scope is None:
                raise ValueError("missing source_id/version requires explicit fallback_scope")
            _require_id(fallback_scope, "fallback_scope")
            digest = hashlib.sha256(content_json.encode("utf-8")).hexdigest()
            key = _json_text(["fingerprint", namespace, fallback_scope, digest])
            source_key = None
            identity = {
                "kind": "fingerprint",
                "stable": False,
                "scope": fallback_scope,
                "sha256": digest,
                "disclosure": "Content fingerprint only; stable source identity/version unavailable.",
            }
        with self._transaction(write=True) as connection:
            revision, visible = self._state(connection, stream_id, create=True)
            self._check_revision(revision, expected_revision)
            previous = connection.execute(
                "SELECT * FROM thread_events WHERE stream_id = ? AND reference_key = ? ORDER BY ordinal DESC LIMIT 1",
                (stream_id, key),
            ).fetchone()
            if previous is not None:
                old_content = self._from_row(previous).payload["content"]
                if _json_text(old_content, canonical=True) != content_json:
                    raise LedgerConflict("reference content changed without a new source version")
            if not force:
                for event in self._events(connection, stream_id, visible):
                    if key == event.reference_key or key in event.represented_reference_keys:
                        return ReferenceAppendResult(self._snapshot(connection, stream_id), False, event, key)
            # A forced read/reinjection of an already seen old version is not
            # a new source revision, and must not supersede a newer observation.
            latest = None
            if source_key is not None:
                for row in connection.execute(
                    "SELECT event_id, reference_key, payload_json FROM thread_events "
                    "WHERE stream_id = ? AND reference_source_key = ? ORDER BY ordinal DESC",
                    (stream_id, source_key),
                ):
                    if json.loads(row["payload_json"]).get("observation", "first_identity") == "first_identity":
                        latest = row
                        break
            supersedes = latest["event_id"] if previous is None and latest is not None else None
            observation = "first_identity" if previous is None else ("explicit_repeat" if force else "reinjection")
            payload = {
                "type": "reference",
                "authority": "reference",
                "reference_key": key,
                "source": {"namespace": namespace, "source_id": source_id, "version": version},
                "identity": identity,
                "content": content,
                "supersedes": supersedes,
                "observation": observation,
            }
            event = LedgerEvent(f"reference:{uuid.uuid4().hex}", "reference", _json_text(payload), key, source_key)
            self._insert_event(connection, stream_id, event)
            self._replace_head(connection, stream_id, revision, [*visible, event.event_id])
            return ReferenceAppendResult(self._snapshot(connection, stream_id), True, event, key)

    @staticmethod
    def _safe_prefix(connection: sqlite3.Connection, stream_id: str, prefix: Sequence[LedgerEvent]) -> None:
        open_calls = set()
        seen_calls = set()
        for event in prefix:
            payload = event.payload
            if isinstance(payload, dict) and (payload.get("role") == "system" or payload.get("toolsAdded")):
                raise UnsafeCompaction("stable system/tool declarations must be preserved outside the compacted prefix")
            for call_id in event.tool_call_ids:
                if call_id in seen_calls:
                    raise UnsafeCompaction(f"duplicate tool call in compaction prefix: {call_id}")
                open_calls.add(call_id)
                seen_calls.add(call_id)
            for call_id in event.tool_result_ids:
                if call_id not in open_calls:
                    raise UnsafeCompaction(f"tool result has no matching prefix call: {call_id}")
                open_calls.remove(call_id)
        if open_calls:
            raise UnsafeCompaction(f"prefix ends before tool results: {sorted(open_calls)!r}")
        for call_id in seen_calls:
            row = connection.execute(
                "SELECT status FROM tool_executions WHERE stream_id = ? AND call_id = ?",
                (stream_id, call_id),
            ).fetchone()
            if row is not None and row["status"] not in ("succeeded", "failed"):
                raise UnsafeCompaction(f"unresolved execution must remain visible: {call_id}")

    def begin_compaction(self, stream_id: str, prefix_count: int, *, start_index: int = 0) -> CompactionPlan:
        with self._transaction() as connection:
            snapshot = self._snapshot(connection, stream_id)
            if type(start_index) is not int or not 0 <= start_index < len(snapshot.events):
                raise ValueError("start_index must leave a nonempty visible prefix")
            if type(prefix_count) is not int or not 0 < prefix_count <= len(snapshot.events) - start_index:
                raise ValueError("prefix_count must select a nonempty visible prefix")
            prefix = snapshot.events[start_index : start_index + prefix_count]
            self._safe_prefix(connection, stream_id, prefix)
            return CompactionPlan(uuid.uuid4().hex, stream_id, snapshot.revision, prefix, snapshot.events[:start_index])

    @staticmethod
    def _plan_json(plan: CompactionPlan) -> str:
        return _json_text(
            {
                "stream_id": plan.stream_id,
                "base_revision": plan.base_revision,
                "leading_ids": [event.event_id for event in plan.leading],
                "prefix": [
                    {
                        "event_id": event.event_id,
                        "kind": event.kind,
                        "payload_json": event.payload_json,
                        "reference_key": event.reference_key,
                        "reference_source_key": event.reference_source_key,
                        "represented_reference_keys": list(event.represented_reference_keys),
                        "tool_call_ids": list(event.tool_call_ids),
                        "tool_result_ids": list(event.tool_result_ids),
                    }
                    for event in plan.prefix
                ],
            }
        )

    def commit_compaction(
        self,
        plan: CompactionPlan,
        summary_payload: Any,
        *,
        retained_reference_keys: Iterable[str] = (),
    ) -> ThreadSnapshot:
        """Commit an exact prefix, preserving any suffix appended while summarizing.

        Only list reference keys actually represented in the summary. The default
        releases every compacted identity for automatic reinjection. Duplicate
        commits with the same plan/output are idempotent, including after restart.
        """
        summary_json = _json_text(summary_payload)
        keys = tuple(sorted(set(retained_reference_keys)))
        plan_json = self._plan_json(plan)
        if not plan.prefix:
            raise ValueError("cannot compact an empty prefix")
        with self._transaction(write=True) as connection:
            committed = connection.execute("SELECT * FROM thread_compactions WHERE token = ?", (plan.token,)).fetchone()
            if committed is not None:
                if (
                    committed["plan_json"] != plan_json
                    or committed["summary_json"] != summary_json
                    or committed["retained_keys_json"] != _json_text(list(keys))
                ):
                    raise LedgerConflict("compaction token reused with different input/output")
                return self._snapshot(connection, plan.stream_id)
            revision, visible = self._state(connection, plan.stream_id)
            start_index = len(plan.leading)
            leading = self._events(connection, plan.stream_id, visible[:start_index])
            current = self._events(connection, plan.stream_id, visible[start_index : start_index + len(plan.prefix)])
            if revision < plan.base_revision or leading != plan.leading or current != plan.prefix:
                raise LedgerConflict("compaction prefix changed; obtain a new snapshot")
            self._safe_prefix(connection, plan.stream_id, current)
            known_keys = {
                key
                for event in current
                for key in (
                    *event.represented_reference_keys,
                    *((event.reference_key,) if event.reference_key else ()),
                )
            }
            if not set(keys).issubset(known_keys):
                raise ValueError("summary cannot retain reference identities absent from its original prefix")
            summary = LedgerEvent(f"compaction:{plan.token}", "summary", summary_json, represented_reference_keys=keys)
            self._insert_event(connection, plan.stream_id, summary)
            self._replace_head(
                connection,
                plan.stream_id,
                revision,
                [*visible[:start_index], summary.event_id, *visible[start_index + len(current) :]],
            )
            connection.execute(
                "INSERT INTO thread_compactions VALUES (?, ?, ?, ?, ?)",
                (plan.token, plan.stream_id, plan_json, summary_json, _json_text(list(keys))),
            )
            return self._snapshot(connection, plan.stream_id)

    @staticmethod
    def _execution(row: Optional[sqlite3.Row]) -> Optional[ToolExecution]:
        return (
            None
            if row is None
            else ToolExecution(
                row["stream_id"],
                row["call_id"],
                row["status"],
                row["intent_json"],
                row["result_event_id"],
                row["result_json"],
            )
        )

    def lookup_tool_execution(self, stream_id: str, call_id: str) -> Optional[ToolExecution]:
        _require_id(stream_id, "stream_id")
        _require_id(call_id, "call_id")
        with self._transaction() as connection:
            return self._execution(
                connection.execute(
                    "SELECT * FROM tool_executions WHERE stream_id = ? AND call_id = ?",
                    (stream_id, call_id),
                ).fetchone()
            )

    def assert_no_unresolved_executions(self, stream_id: str) -> None:
        """Fence every model/tool callback while any external outcome is uncertain.

        A provider may turn an interrupted tool callback into an error toolResult;
        that transport message is not proof that the original effect failed. This
        gate prevents a model from proposing a new call id for the same effect.
        Prepared intents have not been dispatched and are safe to claim normally.
        """
        _require_id(stream_id, "stream_id")
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT call_id, status FROM tool_executions WHERE stream_id = ? "
                "AND status IN ('executing', 'unknown') ORDER BY call_id",
                (stream_id,),
            ).fetchall()
            if rows:
                identities = [(row["call_id"], row["status"]) for row in rows]
                raise LedgerConflict(
                    f"unresolved tool executions require reconciliation before continuing: {identities!r}"
                )

    def prepare_tool_execution(self, stream_id: str, call_id: str, intent: Any) -> ToolExecution:
        """Durably record intent BEFORE attempting any effect; this does not execute it."""
        _require_id(call_id, "call_id")
        intent_json = _json_text(intent)
        with self._transaction(write=True) as connection:
            self._state(connection, stream_id, create=True)
            row = connection.execute(
                "SELECT * FROM tool_executions WHERE stream_id = ? AND call_id = ?", (stream_id, call_id)
            ).fetchone()
            if row is not None and row["intent_json"] != intent_json:
                raise LedgerConflict("tool call id reused with a different execution intent")
            connection.execute(
                "INSERT OR IGNORE INTO tool_executions VALUES (?, ?, ?, 'prepared', NULL, NULL)",
                (stream_id, call_id, intent_json),
            )
            receipt = self._execution(
                connection.execute(
                    "SELECT * FROM tool_executions WHERE stream_id = ? AND call_id = ?",
                    (stream_id, call_id),
                ).fetchone()
            )
            assert receipt is not None
            return receipt

    def claim_tool_execution(self, stream_id: str, call_id: str) -> bool:
        """Only the caller obtaining True may dispatch. Never retry executing/unknown.

        A crash after this durable transition is an uncertain effect, even if the
        process died before dispatch. Reconciliation, not automatic retry, is
        required. Exactly-once delivery to arbitrary external tools is impossible.
        """
        _require_id(stream_id, "stream_id")
        _require_id(call_id, "call_id")
        with self._transaction(write=True) as connection:
            # Close the gap between a callback's read-only fence and its claim.
            # Pi tools are sequential within a stream; child streams remain
            # independently runnable. An uncertain effect cannot be evaded by
            # proposing a fresh call id in the same host thread.
            if (
                connection.execute(
                    "SELECT 1 FROM tool_executions WHERE stream_id = ? AND status IN ('executing', 'unknown') LIMIT 1",
                    (stream_id,),
                ).fetchone()
                is not None
            ):
                return False
            cursor = connection.execute(
                "UPDATE tool_executions SET status = 'executing' WHERE stream_id = ? AND call_id = ? AND status = 'prepared'",
                (stream_id, call_id),
            )
            return cursor.rowcount == 1

    def mark_tool_execution_unknown(
        self,
        stream_id: str,
        call_id: str,
        *,
        observation: Any = None,
    ) -> ToolExecution:
        """Persist uncertainty plus any partial-effect evidence, never a retry grant.

        A failed provider/registry response can still include a sent-message
        receipt. Keep that complete observation for reconciliation without
        declaring either overall success or definite failure. Omitting an
        observation (or passing None) preserves previously recorded evidence.
        """
        _require_id(stream_id, "stream_id")
        _require_id(call_id, "call_id")
        observation_json = None if observation is None else _json_text(observation)
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM tool_executions WHERE stream_id = ? AND call_id = ?", (stream_id, call_id)
            ).fetchone()
            if row is None or row["status"] not in ("executing", "unknown"):
                raise LedgerConflict("only an executing/unknown tool can have an uncertain outcome")
            result_json = row["result_json"] if observation_json is None else observation_json
            connection.execute(
                "UPDATE tool_executions SET status = 'unknown', result_json = ? WHERE stream_id = ? AND call_id = ?",
                (result_json, stream_id, call_id),
            )
            return ToolExecution(stream_id, call_id, "unknown", row["intent_json"], row["result_event_id"], result_json)

    def record_tool_outcome(
        self,
        stream_id: str,
        call_id: str,
        result_payload: Any,
        *,
        status: str = "succeeded",
    ) -> ToolExecution:
        """Save a known callback receipt, without fabricating Pi's native message.

        Pi constructs its exact toolResult after receiving this callback result;
        append that original at message_end. If its ACK is lost, recover using
        this durable receipt rather than dispatching the effect again.
        """
        _require_id(stream_id, "stream_id")
        _require_id(call_id, "call_id")
        if status not in ("succeeded", "failed"):
            raise ValueError("a known execution outcome must be succeeded or failed")
        result_json = _json_text(result_payload)
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM tool_executions WHERE stream_id = ? AND call_id = ?",
                (stream_id, call_id),
            ).fetchone()
            if row is None or row["status"] == "prepared":
                raise LedgerConflict("tool must be claimed before recording its outcome")
            if row["status"] in ("succeeded", "failed"):
                if row["status"] != status or row["result_json"] != result_json:
                    raise LedgerConflict("completed tool outcome is immutable")
                receipt = self._execution(row)
                assert receipt is not None
                return receipt
            connection.execute(
                "UPDATE tool_executions SET status = ?, result_json = ? WHERE stream_id = ? AND call_id = ?",
                (status, result_json, stream_id, call_id),
            )
            return ToolExecution(stream_id, call_id, status, row["intent_json"], result_json=result_json)

    def complete_tool_execution(
        self,
        stream_id: str,
        call_id: str,
        result_event_id: str,
        result_payload: Any,
        *,
        status: str = "succeeded",
    ) -> ThreadSnapshot:
        """Atomically persist a known outcome and its full native result event.

        An unknown execution may be resolved with an authoritative known result,
        but this API never re-executes it. A known failure is terminal too.
        """
        if status not in ("succeeded", "failed"):
            raise ValueError("a known execution outcome must be succeeded or failed")
        inferred_results = _native_tool_ids(result_payload)[1]
        if inferred_results and inferred_results != (call_id,):
            raise LedgerConflict("native tool result call id differs from the execution receipt scope")
        event = _input_event(EventInput(result_event_id, result_payload, tool_result_ids=(call_id,)))
        with self._transaction(write=True) as connection:
            revision, visible = self._state(connection, stream_id)
            row = connection.execute(
                "SELECT * FROM tool_executions WHERE stream_id = ? AND call_id = ?", (stream_id, call_id)
            ).fetchone()
            if row is None or row["status"] == "prepared":
                raise LedgerConflict("tool must be claimed before recording its outcome")
            if row["status"] in ("succeeded", "failed"):
                if (
                    row["status"] != status
                    or row["result_event_id"] != result_event_id
                    or row["result_json"] != event.payload_json
                ):
                    raise LedgerConflict("completed tool outcome is immutable")
                return self._snapshot(connection, stream_id)
            if self._insert_event(connection, stream_id, event):
                self._replace_head(connection, stream_id, revision, [*visible, event.event_id])
            connection.execute(
                "UPDATE tool_executions SET status = ?, result_event_id = ?, result_json = ? "
                "WHERE stream_id = ? AND call_id = ?",
                (status, result_event_id, event.payload_json, stream_id, call_id),
            )
            return self._snapshot(connection, stream_id)
