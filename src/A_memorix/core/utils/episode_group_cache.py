"""Persist completed segmentation groups before a whole source can be published."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import json
import sqlite3
import time
import threading


@dataclass
class EpisodeCallBudget:
    limit: int
    used: int = 0

    def reserve(self) -> None:
        if self.used >= self.limit:
            raise RuntimeError("episode_model_call_budget_exhausted")
        self.used += 1


class EpisodeGroupCache:
    """Derived results only; source evidence remains in metadata.db."""

    def __init__(self, path: Path):
        self.path = path
        self._connection = None
        self._lock = threading.RLock()

    @contextmanager
    def _connect(self):
        with self._lock:
            if self._connection is None:
                connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("""CREATE TABLE IF NOT EXISTS groups (
                    source TEXT NOT NULL, data_fingerprint TEXT NOT NULL,
                    generation_hash TEXT NOT NULL, input_fingerprint TEXT NOT NULL,
                    paragraph_ids TEXT NOT NULL, payloads TEXT NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(source, data_fingerprint, generation_hash))""")
                self._connection = connection
            yield self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def list_source(self, source: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            return [
                {
                    **dict(row),
                    "paragraph_ids": json.loads(row["paragraph_ids"]),
                    "payloads": json.loads(row["payloads"]),
                }
                for row in connection.execute("SELECT * FROM groups WHERE source=? ORDER BY updated_at DESC", (source,))
            ]

    def put(
        self,
        source: str,
        data_fingerprint: str,
        generation_hash: str,
        input_fingerprint: str,
        paragraph_ids: List[str],
        payloads: List[Dict[str, Any]],
    ) -> None:
        with self._connect() as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO groups VALUES (?,?,?,?,?,?,?)",
                (
                    source,
                    data_fingerprint,
                    generation_hash,
                    input_fingerprint,
                    json.dumps(paragraph_ids),
                    json.dumps(payloads, ensure_ascii=False),
                    time.time(),
                ),
            )

    def retain(self, source: str, fingerprints: List[str]) -> None:
        with self._connect() as connection, connection:
            connection.execute(
                "DELETE FROM groups WHERE source=? AND data_fingerprint NOT IN (SELECT value FROM json_each(?))",
                (source, json.dumps(fingerprints)),
            )
        self.close()
