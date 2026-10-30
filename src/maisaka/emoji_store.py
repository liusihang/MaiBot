"""Durable, session-scoped emoji resource IDs, not a second model transcript.

Pi may retain a collage across process restarts. Its visible integer must therefore
keep referring to the same business resource, independently of Python chat history.
Legacy process-local low numbers are intentionally not imported or guessed.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Union

import sqlite3

from src.common.paths import get_instance_root

# Reserve the former process-local low-number namespace. An arbitrary unknown
# legacy integer cannot be proven collision-free, but known ordinary low IDs are
# rejected, rather than being silently rebound to the first newly shown collage.
FIRST_DURABLE_EMOJI_INDEX = 1_000_000_000
MAX_EMOJI_INDEX = (1 << 53) - 1  # Exact in both Python and Pi/JavaScript JSON.


def _require_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Emoji candidates require a real nonempty session_id")


class EmojiCandidateStore:
    """A small immutable number-to-hash map shared across loops and processes.

    Construction performs no I/O. Call ``allocate`` and ``resolve`` in a worker
    thread. No connection or event-loop-affine object is kept on this instance.
    Entries never expire automatically: an old Pi history may still reference one.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        path = self._db_path
        if path is None:
            path = get_instance_root() / "data/pi_harness/emoji_candidates.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=10, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            # Serialize schema initialization and whole-batch allocation, including
            # cold concurrent starts in separate bot/WebUI processes.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS emoji_candidates ("
                "candidate_index INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, emoji_hash TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO sqlite_sequence(name, seq) SELECT 'emoji_candidates', ? "
                "WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name='emoji_candidates')",
                (FIRST_DURABLE_EMOJI_INDEX - 1,),
            )
            connection.execute(
                "UPDATE sqlite_sequence SET seq=? WHERE name='emoji_candidates' AND seq<?",
                (FIRST_DURABLE_EMOJI_INDEX - 1, FIRST_DURABLE_EMOJI_INDEX - 1),
            )
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def allocate(self, session_id: str, emoji_hashes: Sequence[str]) -> Dict[int, str]:
        """Commit a contiguous batch before its collage can become model-visible."""
        _require_session_id(session_id)
        hashes = list(emoji_hashes)
        if any(not isinstance(value, str) or not value.strip() for value in hashes):
            raise ValueError("Emoji candidates require nonempty resource hashes")
        if not hashes:
            return {}
        with self._transaction() as connection:
            current = connection.execute("SELECT seq FROM sqlite_sequence WHERE name='emoji_candidates'").fetchone()[0]
            if current + len(hashes) > MAX_EMOJI_INDEX:
                raise RuntimeError("Durable emoji index namespace exhausted")
            result: Dict[int, str] = {}
            for emoji_hash in hashes:
                cursor = connection.execute(
                    "INSERT INTO emoji_candidates(session_id, emoji_hash) VALUES (?, ?)",
                    (session_id, emoji_hash),
                )
                result[int(cursor.lastrowid)] = emoji_hash
            return result

    def resolve(self, session_id: str, candidate_index: int) -> Optional[str]:
        """Look up only this session's immutable mapping; never inspect chat history."""
        _require_session_id(session_id)
        if type(candidate_index) is not int or not FIRST_DURABLE_EMOJI_INDEX <= candidate_index <= MAX_EMOJI_INDEX:
            return None
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT emoji_hash FROM emoji_candidates WHERE session_id=? AND candidate_index=?",
                (session_id, candidate_index),
            ).fetchone()
            return row[0] if row is not None else None
