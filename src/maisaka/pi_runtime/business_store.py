"""Business inbox and delivery receipts; Pi SessionManager alone owns conversation history.

No model transcript, visible head, compaction, or replay reconstruction lives here.
An acknowledged inbox row is only an idempotency receipt for a source message.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import json
import sqlite3
import uuid


class LedgerConflict(RuntimeError):
    """A source identity or delivery outcome conflicts with its durable receipt."""


def _require_id(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty identifier")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class BusinessInput:
    event_id: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ToolExecution:
    stream_id: str
    call_id: str
    status: str
    intent_json: str
    result_event_id: Optional[str] = None
    result_json: Optional[str] = None

    @property
    def result(self) -> Any:
        return None if self.result_json is None else json.loads(self.result_json)


class BusinessStore:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS business_inbox (
                    ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(stream_id,event_id)
                );
                CREATE TABLE IF NOT EXISTS business_turns (
                    stream_id TEXT PRIMARY KEY, turn_id TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS business_sources (
                    stream_id TEXT NOT NULL, event_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    PRIMARY KEY(stream_id,event_id)
                );
                CREATE TABLE IF NOT EXISTS business_tool_media (
                    stream_id TEXT NOT NULL, event_id TEXT NOT NULL, call_id TEXT NOT NULL,
                    PRIMARY KEY(stream_id,event_id)
                );
                CREATE TABLE IF NOT EXISTS tool_executions (
                    stream_id TEXT NOT NULL, call_id TEXT NOT NULL,
                    intent_json TEXT NOT NULL, status TEXT NOT NULL,
                    result_event_id TEXT, result_json TEXT,
                    PRIMARY KEY(stream_id,call_id),
                    CHECK(status IN ('prepared','executing','unknown','succeeded','failed'))
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
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

    def validate_source(self, stream_id: str, event_id: str, fingerprint: str) -> None:
        """Validate adapter source identity even after its inbox was acknowledged.

        This fingerprint is immutable business input, not a regenerated model
        rendering whose aliases or media annotations may change later.
        """
        _require_id(stream_id, "stream_id")
        _require_id(event_id, "event_id")
        _require_id(fingerprint, "fingerprint")
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT fingerprint FROM business_sources WHERE stream_id=? AND event_id=?",
                (stream_id, event_id),
            ).fetchone()
            if row is not None and row[0] != fingerprint:
                raise LedgerConflict("Source ID reused with changed original content")
            connection.execute(
                "INSERT OR IGNORE INTO business_sources VALUES(?,?,?)", (stream_id, event_id, fingerprint)
            )

    def is_tool_media_input(self, stream_id: str, event_id: str) -> bool:
        """A host UI media snapshot already belongs to a durable SDK toolResult."""
        with self._transaction() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM business_tool_media WHERE stream_id=? AND event_id=?", (stream_id, event_id)
                ).fetchone()
                is not None
            )

    def has_input(self, stream_id: str, event_id: str) -> bool:
        with self._transaction() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM business_inbox WHERE stream_id=? AND event_id=?",
                    (stream_id, event_id),
                ).fetchone()
                is not None
            )

    def enqueue_input(self, stream_id: str, event_id: str, payload: Dict[str, Any]) -> bool:
        _require_id(stream_id, "stream_id")
        _require_id(event_id, "event_id")
        encoded = _json_text(payload)
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM business_inbox WHERE stream_id=? AND event_id=?",
                (stream_id, event_id),
            ).fetchone()
            if row is not None:
                if row[0] != encoded:
                    raise LedgerConflict("Source ID reused with changed input")
                return False
            connection.execute(
                "INSERT INTO business_inbox(stream_id,event_id,payload_json) VALUES(?,?,?)",
                (stream_id, event_id, encoded),
            )
            return True

    def pending_inputs(self, stream_id: str) -> List[BusinessInput]:
        with self._transaction() as connection:
            return [
                BusinessInput(row[0], json.loads(row[1]))
                for row in connection.execute(
                    "SELECT event_id,payload_json FROM business_inbox WHERE stream_id=? AND acknowledged=0 ORDER BY ordinal",
                    (stream_id,),
                )
            ]

    def pending_turn(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Selected business work is independent of delivery into model history."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT turn_id,source_ids_json FROM business_turns WHERE stream_id=? AND completed=0",
                (stream_id,),
            ).fetchone()
            return None if row is None else {"turn_id": row[0], "source_ids": json.loads(row[1])}

    def begin_turn(self, stream_id: str, source_ids: List[str]) -> Dict[str, Any]:
        _require_id(stream_id, "stream_id")
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT turn_id,source_ids_json FROM business_turns WHERE stream_id=? AND completed=0",
                (stream_id,),
            ).fetchone()
            turn_id = row[0] if row else uuid.uuid4().hex
            sources = list(dict.fromkeys([*(json.loads(row[1]) if row else []), *source_ids]))
            connection.execute(
                "INSERT INTO business_turns VALUES(?,?,?,0) ON CONFLICT(stream_id) DO UPDATE SET "
                "turn_id=excluded.turn_id,source_ids_json=excluded.source_ids_json,completed=0",
                (stream_id, turn_id, _json_text(sources)),
            )
            return {"turn_id": turn_id, "source_ids": sources}

    def complete_turn(self, stream_id: str, turn_id: str) -> None:
        with self._transaction(write=True) as connection:
            changed = connection.execute(
                "UPDATE business_turns SET completed=1 WHERE stream_id=? AND turn_id=?",
                (stream_id, turn_id),
            ).rowcount
            if changed != 1:
                raise LedgerConflict("Completed turn does not match selected work")

    def recovery_streams(self) -> List[str]:
        """Only selected unfinished work or undelivered inbox rows need startup."""
        with self._transaction() as connection:
            return [row[0] for row in connection.execute(
                "SELECT stream_id FROM business_turns WHERE completed=0 UNION "
                "SELECT stream_id FROM business_inbox WHERE acknowledged=0 ORDER BY stream_id"
            )]

    def acknowledge_input(self, stream_id: str, event_id: str) -> None:
        """Call only after Pi has durably recorded this exact source ID.

        A lost ACK leaves the row pending. Re-delivery is deduplicated by Pi's
        own source marker in SessionManager before acknowledging it here.
        """
        with self._transaction(write=True) as connection:
            cursor = connection.execute(
                "UPDATE business_inbox SET acknowledged=1 WHERE stream_id=? AND event_id=?",
                (stream_id, event_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(event_id)

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
            _require_id(stream_id, "stream_id")
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
            for event_id in result_payload.get("details", {}).get("hostMediaEventIds", []):
                _require_id(event_id, "media event_id")
                previous = connection.execute(
                    "SELECT call_id FROM business_tool_media WHERE stream_id=? AND event_id=?", (stream_id, event_id)
                ).fetchone()
                if previous is not None and previous[0] != call_id:
                    raise LedgerConflict("Host media snapshot cannot be rebound to another tool call")
                connection.execute(
                    "INSERT OR IGNORE INTO business_tool_media VALUES(?,?,?)", (stream_id, event_id, call_id)
                )
            connection.execute(
                "UPDATE tool_executions SET status = ?, result_json = ? WHERE stream_id = ? AND call_id = ?",
                (status, result_json, stream_id, call_id),
            )
            return ToolExecution(stream_id, call_id, status, row["intent_json"], result_json=result_json)
