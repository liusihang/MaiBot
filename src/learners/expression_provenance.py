"""Small durable source association; no copied chat text or invented legacy examples."""

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
            db.execute(
                "INSERT OR REPLACE INTO expression_sources VALUES (?,?,?,?)",
                (expression_id, identity, session_id, json.dumps(sources, ensure_ascii=False)),
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
