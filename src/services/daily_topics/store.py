"""Durable, session-isolated topic sources, resumable jobs, and immutable boards.

This database is independent of chat/model histories. Every method opens its own
connection so bot/WebUI event loops and separate workers may safely share it.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import hashlib
import json
import math
import sqlite3
import uuid


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class LostLease(RuntimeError):
    """Another worker owns the session, or this worker's lease has expired."""


class TopicStore:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("TopicStore requires a durable file, not :memory:")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sources (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, source_id TEXT NOT NULL,
                    occurred_at REAL NOT NULL, author TEXT NOT NULL, payload TEXT NOT NULL,
                    covered_day TEXT, UNIQUE(session_id, source_id)
                );
                CREATE INDEX IF NOT EXISTS sources_pending
                    ON sources(session_id, covered_day, occurred_at, seq);
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, lease_token TEXT, lease_until REAL,
                    last_completed_day TEXT
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    session_id TEXT NOT NULL, cycle_day TEXT NOT NULL, cutoff REAL NOT NULL,
                    watermark INTEGER NOT NULL, batch_chars INTEGER NOT NULL,
                    next_batch INTEGER NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 0,
                    topics_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    published_version INTEGER, released_cycle_day TEXT, changed INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(session_id, cycle_day)
                );
                CREATE TABLE IF NOT EXISTS job_sources (
                    session_id TEXT NOT NULL, cycle_day TEXT NOT NULL, source_seq INTEGER NOT NULL,
                    PRIMARY KEY(session_id, cycle_day, source_seq),
                    FOREIGN KEY(source_seq) REFERENCES sources(seq)
                );
                CREATE TABLE IF NOT EXISTS versions (
                    session_id TEXT NOT NULL, version INTEGER NOT NULL, as_of_day TEXT NOT NULL,
                    topics_json TEXT NOT NULL, content_hash TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(session_id, version), UNIQUE(session_id, as_of_day)
                );
            """)

    @contextmanager
    def _connection(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def import_cursor(self) -> int:
        with self._connection() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='import_cursor'").fetchone()
            return int(row[0]) if row else 0

    def import_anchor(self) -> Optional[Dict[str, Any]]:
        with self._connection() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='import_anchor'").fetchone()
            return json.loads(row[0]) if row else None

    def initial_boundary(self) -> Optional[Dict[str, Any]]:
        with self._connection() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='initial_boundary'").fetchone()
            return json.loads(row[0]) if row else None

    def initialize_import(self, cursor: int, boundary: Dict[str, Any]) -> Dict[str, Any]:
        """Record the explicit initial history policy once; never rewind imports."""
        if type(cursor) is not int or cursor < 0:
            raise ValueError("cursor must be a nonnegative integer")
        encoded = canonical_json(boundary)
        with self._connection(write=True) as db:
            existing = db.execute("SELECT value FROM metadata WHERE key='initial_boundary'").fetchone()
            if existing:
                return json.loads(existing[0])
            db.execute("INSERT INTO metadata VALUES ('initial_boundary', ?)", (encoded,))
            db.execute(
                "INSERT INTO metadata VALUES ('import_cursor', ?) ON CONFLICT(key) "
                "DO UPDATE SET value=CAST(MAX(CAST(value AS INTEGER), ?) AS TEXT)",
                (str(cursor), cursor),
            )
        return boundary

    def ingest_page(self, records: List[Dict[str, Any]], cursor: int, anchor: Optional[Dict[str, Any]] = None) -> int:
        """Atomically ingest a keyset page and its cursor, including empty pages.

        Source text is immutable. Replayed identical rows are harmless; a reused
        source ID with different contents fails rather than silently losing edits.
        """
        if type(cursor) is not int or cursor < 0:
            raise ValueError("cursor must be a nonnegative integer")
        if anchor is not None and (anchor.get("row_id") != cursor or not isinstance(anchor.get("fingerprint"), str)):
            raise ValueError("import anchor must identify the page cursor row")
        prepared = []
        for record in records:
            for key in ("source_id", "message_id", "session_id", "author", "text"):
                if not isinstance(record.get(key), str):
                    raise ValueError(f"source {key} must be a string")
            if not record["source_id"] or not record["session_id"]:
                raise ValueError("source and session IDs must not be empty")
            stamp = record.get("occurred_at")
            if type(stamp) not in (int, float) or not math.isfinite(stamp):
                raise ValueError("occurred_at must be a finite epoch timestamp")
            prepared.append((record, canonical_json(record)))
        inserted = 0
        with self._connection(write=True) as db:
            for record, payload in prepared:
                old = db.execute(
                    "SELECT payload FROM sources WHERE session_id=? AND source_id=?",
                    (record["session_id"], record["source_id"]),
                ).fetchone()
                if old:
                    if old[0] != payload:
                        raise ValueError(f"immutable source changed: {record['source_id']}")
                    continue
                db.execute(
                    "INSERT INTO sources(session_id,source_id,occurred_at,author,payload) VALUES (?,?,?,?,?)",
                    (record["session_id"], record["source_id"], record["occurred_at"], record["author"], payload),
                )
                inserted += 1
            db.execute(
                "INSERT INTO metadata VALUES ('import_cursor', ?) ON CONFLICT(key) "
                "DO UPDATE SET value=CAST(MAX(CAST(value AS INTEGER), ?) AS TEXT)",
                (str(cursor), cursor),
            )
            if anchor is not None:
                # A slower concurrent importer must not overwrite a newer anchor.
                saved = int(db.execute("SELECT value FROM metadata WHERE key='import_cursor'").fetchone()[0])
                if cursor == saved:
                    existing_anchor = db.execute("SELECT value FROM metadata WHERE key='import_anchor'").fetchone()
                    if existing_anchor:
                        old_anchor = json.loads(existing_anchor[0])
                        if old_anchor["row_id"] == cursor and old_anchor != anchor:
                            raise ValueError("import cursor row fingerprint changed; source database may have reset")
                    db.execute(
                        "INSERT INTO metadata VALUES ('import_anchor', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (canonical_json(anchor),),
                    )
        return inserted

    def sessions(self) -> List[str]:
        with self._connection() as db:
            rows = db.execute("SELECT session_id FROM sources UNION SELECT session_id FROM sessions ORDER BY 1")
            return [row[0] for row in rows]

    @staticmethod
    def _current(db: sqlite3.Connection, session_id: str) -> Optional[Dict[str, Any]]:
        row = db.execute(
            "SELECT * FROM versions WHERE session_id=? ORDER BY version DESC LIMIT 1", (session_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["topics"] = json.loads(result.pop("topics_json"))
        return result

    def current(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as db:
            return self._current(db, session_id)

    def coverage(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT seq,source_id,occurred_at,covered_day FROM sources WHERE session_id=? ORDER BY seq",
                    (session_id,),
                )
            ]

    def read_sources(self, session_id: str, source_ids: List[str]) -> List[Dict[str, Any]]:
        """Read immutable evidence only within the caller's real session."""
        if not source_ids:
            return []
        if len(source_ids) > 100:
            raise ValueError("read at most 100 source IDs per request")
        placeholders = ",".join("?" for _ in source_ids)
        with self._connection() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    f"SELECT payload FROM sources WHERE session_id=? AND source_id IN ({placeholders}) ORDER BY seq",
                    [session_id, *source_ids],
                )
            ]

    def read_evidence_metadata(self, session_id: str, source_ids: List[str]) -> List[Dict[str, Any]]:
        """Actual event times for prior claims, without reloading long raw text."""
        if not source_ids:
            return []
        if len(source_ids) > 100:
            raise ValueError("read at most 100 source IDs per request")
        placeholders = ",".join("?" for _ in source_ids)
        with self._connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    f"SELECT source_id,occurred_at,author FROM sources WHERE session_id=? AND source_id IN ({placeholders})",
                    [session_id, *source_ids],
                )
            ]

    @staticmethod
    def _job(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        result["topics"] = json.loads(result.pop("topics_json"))
        result["changed"] = bool(result["changed"])
        return result

    def job_status(self, session_id: str, cycle_day: str) -> Optional[Dict[str, Any]]:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE session_id=? AND cycle_day=?", (session_id, cycle_day)
            ).fetchone()
            return self._job(row) if row else None

    def acquire(
        self, session_id: str, cycle_day: str, cutoff: float, now: float, lease_seconds: float, batch_chars: int
    ) -> Dict[str, Any]:
        """Lease a session and freeze its exact eligible source set on first use."""
        with self._connection(write=True) as db:
            db.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES (?)", (session_id,))
            session = db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if session["lease_token"] and session["lease_until"] > now:
                return {"status": "busy", "cycle_day": cycle_day}
            if session["last_completed_day"] and cycle_day <= session["last_completed_day"]:
                completed = db.execute(
                    "SELECT * FROM jobs WHERE session_id=? AND released_cycle_day=? AND status='complete'",
                    (session_id, cycle_day),
                ).fetchone()
                if completed:
                    return {"status": "complete", "job": self._job(completed), "cycle_day": completed["cycle_day"]}
                return {"status": "stale", "cycle_day": cycle_day}
            pending = db.execute(
                "SELECT * FROM jobs WHERE session_id=? AND status='pending' ORDER BY cycle_day LIMIT 1",
                (session_id,),
            ).fetchone()
            if pending:
                if pending["cycle_day"] > cycle_day:
                    return {"status": "stale", "cycle_day": cycle_day}
                cycle_day = pending["cycle_day"]
                job = pending
            else:
                job = db.execute(
                    "SELECT * FROM jobs WHERE session_id=? AND cycle_day=?", (session_id, cycle_day)
                ).fetchone()
                if job:
                    return {"status": "complete", "job": self._job(job), "cycle_day": cycle_day}
                current = self._current(db, session_id)
                watermark = db.execute(
                    "SELECT COALESCE(MAX(seq),0) FROM sources WHERE session_id=? AND covered_day IS NULL "
                    "AND occurred_at<?",
                    (session_id, cutoff),
                ).fetchone()[0]
                db.execute(
                    "INSERT INTO jobs(session_id,cycle_day,cutoff,watermark,batch_chars,topics_json) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        session_id,
                        cycle_day,
                        cutoff,
                        watermark,
                        batch_chars,
                        canonical_json(current["topics"] if current else []),
                    ),
                )
                db.execute(
                    "INSERT INTO job_sources SELECT session_id,?,seq FROM sources WHERE session_id=? "
                    "AND covered_day IS NULL AND occurred_at<? AND seq<=?",
                    (cycle_day, session_id, cutoff, watermark),
                )
                job = db.execute(
                    "SELECT * FROM jobs WHERE session_id=? AND cycle_day=?", (session_id, cycle_day)
                ).fetchone()
            token = uuid.uuid4().hex
            db.execute(
                "UPDATE sessions SET lease_token=?,lease_until=? WHERE session_id=?",
                (token, now + lease_seconds, session_id),
            )
            sources = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT s.payload FROM sources s JOIN job_sources j ON j.source_seq=s.seq "
                    "WHERE j.session_id=? AND j.cycle_day=? ORDER BY s.occurred_at,s.seq",
                    (session_id, cycle_day),
                )
            ]
            return {
                "status": "working",
                "job": self._job(job),
                "token": token,
                "sources": sources,
                "cycle_day": cycle_day,
                "previous_as_of_day": (self._current(db, session_id) or {}).get("as_of_day"),
            }

    @staticmethod
    def _assert_lease(db: sqlite3.Connection, session_id: str, token: str, now: float) -> None:
        row = db.execute("SELECT lease_token,lease_until FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None or row[0] != token or row[1] <= now:
            raise LostLease("daily-topic session lease lost")

    def renew(self, session_id: str, token: str, now: float, lease_seconds: float) -> None:
        with self._connection(write=True) as db:
            self._assert_lease(db, session_id, token, now)
            db.execute("UPDATE sessions SET lease_until=? WHERE session_id=?", (now + lease_seconds, session_id))

    def release(self, session_id: str, token: str) -> None:
        with self._connection(write=True) as db:
            db.execute(
                "UPDATE sessions SET lease_token=NULL,lease_until=NULL WHERE session_id=? AND lease_token=?",
                (session_id, token),
            )

    def stage(
        self, session_id: str, cycle_day: str, token: str, revision: int, topics: List[Dict[str, Any]], now: float
    ) -> None:
        with self._connection(write=True) as db:
            self._assert_lease(db, session_id, token, now)
            changed = db.execute(
                "UPDATE jobs SET topics_json=?,next_batch=next_batch+1,revision=revision+1 "
                "WHERE session_id=? AND cycle_day=? AND revision=? AND status='pending'",
                (canonical_json(topics), session_id, cycle_day, revision),
            ).rowcount
            if changed != 1:
                raise LostLease("daily-topic job revision changed")

    def finish(
        self,
        session_id: str,
        cycle_day: str,
        token: str,
        revision: int,
        total_batches: int,
        now: float,
        released_cycle_day: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish the final board and cover all frozen sources in one commit."""
        released_cycle_day = released_cycle_day or cycle_day
        if released_cycle_day < cycle_day:
            raise ValueError("release day cannot precede the job cycle")
        with self._connection(write=True) as db:
            self._assert_lease(db, session_id, token, now)
            last_day = db.execute(
                "SELECT last_completed_day FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            if last_day and last_day >= released_cycle_day:
                raise LostLease("daily-topic release slot has already completed")
            row = db.execute(
                "SELECT * FROM jobs WHERE session_id=? AND cycle_day=?", (session_id, cycle_day)
            ).fetchone()
            if row["status"] != "pending" or row["revision"] != revision or row["next_batch"] != total_batches:
                raise LostLease("daily-topic job is not ready to publish")
            current = self._current(db, session_id)
            digest = hashlib.sha256(row["topics_json"].encode("utf-8")).hexdigest()
            previous_hash = current["content_hash"] if current else hashlib.sha256(b"[]").hexdigest()
            changed = digest != previous_hash
            version = current["version"] if current else None
            if changed:
                version = (version or 0) + 1
                db.execute(
                    "INSERT INTO versions VALUES (?,?,?,?,?,?)",
                    (session_id, version, released_cycle_day, row["topics_json"], digest, now),
                )
            db.execute(
                "UPDATE sources SET covered_day=? WHERE seq IN "
                "(SELECT source_seq FROM job_sources WHERE session_id=? AND cycle_day=?) AND covered_day IS NULL",
                (cycle_day, session_id, cycle_day),
            )
            db.execute(
                "UPDATE jobs SET status='complete',published_version=?,released_cycle_day=?,changed=? WHERE session_id=? AND cycle_day=?",
                (version, released_cycle_day, int(changed), session_id, cycle_day),
            )
            db.execute(
                "UPDATE sessions SET last_completed_day=?,lease_token=NULL,lease_until=NULL WHERE session_id=?",
                (released_cycle_day, session_id),
            )
            return {
                "status": "complete",
                "cycle_day": cycle_day,
                "published_cycle_day": released_cycle_day,
                "version": version,
                "changed": changed,
            }
