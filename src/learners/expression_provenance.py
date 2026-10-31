"""Durable expression sources and authentic example prefixes from learning messages."""

from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List

import hashlib
import json
import sqlite3


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def expression_identity(situation: str, style: str, created_at: str) -> str:
    return text_digest(json.dumps([situation, style, created_at], ensure_ascii=False))


class ExpressionProvenance:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, expression_id: int, identity: str, session_id: str, sources: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS expression_sources (
                expression_id INTEGER NOT NULL, identity TEXT NOT NULL, session_id TEXT NOT NULL,
                sources TEXT NOT NULL, PRIMARY KEY(expression_id, session_id))""")
            previous = db.execute(
                "SELECT identity,sources FROM expression_sources WHERE expression_id=? AND session_id=?",
                (expression_id, session_id),
            ).fetchone()
            old_sources = json.loads(previous[1]) if previous and previous[0] == identity else []
            unique = {}
            for source in [*sources, *old_sources]:
                unique.setdefault((source["platform"], source["message_id"]), source)
            db.execute(
                "INSERT OR REPLACE INTO expression_sources VALUES (?,?,?,?)",
                (expression_id, identity, session_id, json.dumps(list(unique.values())[:3], ensure_ascii=False)),
            )

    def load(self, session_id: str) -> Dict[int, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='expression_sources'").fetchone():
                return {}
            return {
                row[0]: {"identity": row[1], "sources": json.loads(row[2])}
                for row in db.execute(
                    "SELECT expression_id,identity,sources FROM expression_sources WHERE session_id=?", (session_id,)
                )
            }
