"""Durable source receipts and reusable model results for joint learning."""

from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

import hashlib
import json
import sqlite3

INTERVAL_SECONDS = 30 * 60
BATCH_SIZE = 100
RETRY_SECONDS = 300


class JointLearningJournal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS sources (
                session_id TEXT NOT NULL, message_id TEXT NOT NULL, queued REAL NOT NULL,
                batch_id TEXT, PRIMARY KEY(session_id,message_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS batches (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source_ids TEXT NOT NULL,
                extraction TEXT, review TEXT, expressions_done INTEGER NOT NULL DEFAULT 0,
                jargon_done INTEGER NOT NULL DEFAULT 0, done INTEGER NOT NULL DEFAULT 0,
                retry_at REAL NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0)""")
            if "attempts" not in {row[1] for row in db.execute("PRAGMA table_info(batches)")}:
                db.execute("ALTER TABLE batches ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
            db.execute("CREATE INDEX IF NOT EXISTS sources_pending ON sources(session_id,batch_id,queued)")

    def enqueue(self, session_id: str, ids: List[str], now: float) -> List[str]:
        with closing(self.connect()) as db, db:
            added = []
            for mid in ids:
                result = db.execute(
                    "INSERT OR IGNORE INTO sources(session_id,message_id,queued) VALUES (?,?,?)", (session_id, mid, now)
                )
                if result.rowcount:
                    added.append(mid)
            return added

    def next_batch(self, session_id: str, now: float) -> Optional[Dict[str, Any]]:
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM batches WHERE session_id=? AND done=0 ORDER BY rowid LIMIT 1", (session_id,)
            ).fetchone()
            if existing:
                return dict(existing) if existing["retry_at"] <= now else None
            rows = db.execute(
                "SELECT message_id,queued FROM sources WHERE session_id=? AND batch_id IS NULL "
                "ORDER BY queued,rowid LIMIT ?",
                (session_id, BATCH_SIZE),
            ).fetchall()
            if not rows or (len(rows) < BATCH_SIZE and rows[0]["queued"] + INTERVAL_SECONDS > now):
                return None
            ids = [row["message_id"] for row in rows]
            batch_id = hashlib.sha256(json.dumps([session_id, ids]).encode()).hexdigest()
            db.execute(
                "INSERT INTO batches(id,session_id,source_ids) VALUES (?,?,?)", (batch_id, session_id, json.dumps(ids))
            )
            db.executemany(
                "UPDATE sources SET batch_id=? WHERE session_id=? AND message_id=?",
                [(batch_id, session_id, mid) for mid in ids],
            )
            return dict(db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone())

    def save(self, batch_id: str, **fields: Any) -> None:
        allowed = {"extraction", "review", "expressions_done", "jargon_done", "done", "retry_at"}
        if not fields or not set(fields).issubset(allowed):
            raise ValueError("Invalid learning checkpoint fields")
        with closing(self.connect()) as db, db:
            db.execute(
                "UPDATE batches SET " + ",".join(f"{key}=?" for key in fields) + " WHERE id=?",
                [*fields.values(), batch_id],
            )

    def fail(self, batch_id: str, now: float) -> bool:
        """Retain failed evidence for inspection, but stop spending after three attempts."""
        with closing(self.connect()) as db, db:
            db.execute(
                "UPDATE batches SET attempts=attempts+1, retry_at=?, "
                "done=CASE WHEN attempts+1>=3 THEN 2 ELSE 0 END WHERE id=?",
                (now + RETRY_SECONDS, batch_id),
            )
            return db.execute("SELECT done FROM batches WHERE id=?", (batch_id,)).fetchone()[0] == 2
