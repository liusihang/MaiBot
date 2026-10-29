"""Durable public tool evidence, separate from opaque provider replay packets."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import asyncio
import json
import sqlite3

from src.maisaka.tool_result_store import ToolResultStore, tool_result_store


READ_THREAD_ARCHIVE_TOOL_DEFINITION = {"type": "function", "function": {
    "name": "read_thread_archive",
    "description": "Read public original chat, reference and tool evidence from archived events in this chat. Provider replay and private reasoning are not exposed.",
    "parameters": {"type": "object", "properties": {
        "event_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20},
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 2800},
    }, "required": ["event_ids"], "additionalProperties": False},
}}


def _page(text: str, offset: int, limit: int, keyword: str = "") -> Dict[str, Any]:
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 2800:
        raise ValueError("Invalid evidence pagination")
    if len(keyword) > 256:
        raise ValueError("Evidence keyword is too long")
    start = min(offset, len(text))
    if keyword:
        position = text.find(keyword, start)
        if position < 0:
            return {"text": "", "offset": start, "next_offset": None, "total_chars": len(text), "matched": False}
        start = max(start, position - min(250, limit // 4))
    end = min(start + limit, len(text))
    return {"text": text[start:end], "offset": start, "next_offset": end if end < len(text) else None,
            "total_chars": len(text), "matched": True}


def public_archive_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Explicit allowlist, never recurse into native/provider/private message fields."""
    role = payload.get("role")
    if role == "system":
        return {"type": "internal_system_record", "text": "[System and tool configuration omitted]"}
    if role in {"user", "assistant", "toolResult"}:
        content = payload.get("content", [])
        text = content if isinstance(content, str) else "\n".join(
            item["text"] for item in content if item.get("type") == "text"
        )
        return {"type": str(role), "text": text, "tool_name": payload.get("toolName", ""),
                "call_id": payload.get("toolCallId", ""), "source_message_id": payload.get("sourceMessageId", ""),
                "source_archive": payload.get("sourceArchive", {}) if payload.get("kind") == "summary" else {}}
    if payload.get("type") == "reference":
        return {"type": "reference", "source": payload.get("source", {}), "content": payload.get("content", "")}
    return {"type": "internal_record", "text": "[No public text in this event]"}


class DurableEvidenceStore:
    """Promote an existing scoped raw result before its bounded memory store evicts it."""

    def __init__(self, db_path: str, source_store: ToolResultStore = tool_result_store) -> None:
        self.db_path = str(db_path)
        self.source_store = source_store

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("""CREATE TABLE IF NOT EXISTS pi_durable_evidence(
            parent TEXT NOT NULL, handle TEXT NOT NULL, child TEXT NOT NULL,
            tool_name TEXT NOT NULL, raw TEXT NOT NULL, metadata TEXT NOT NULL,
            PRIMARY KEY(parent,handle))""")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _save(self, parent: str, handle: str, child: str, tool: str, raw: str, metadata: Dict[str, Any]) -> None:
        with self._connection() as connection:
            old = connection.execute("SELECT * FROM pi_durable_evidence WHERE parent=? AND handle=?", (parent, handle)).fetchone()
            if old is not None:
                if old["raw"] != raw or old["child"] != child or old["tool_name"] != tool:
                    raise ValueError("Evidence handle cannot be rebound to different content or ownership")
                return
            connection.execute("INSERT INTO pi_durable_evidence VALUES(?,?,?,?,?,?)", (
                parent, handle, child, tool, raw, json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ))

    async def capture(
        self, parent_session_id: str, result: Any, *, child_namespace: str = "",
        source_session_id: Optional[str] = None,
    ) -> Optional[str]:
        if not parent_session_id.strip():
            raise ValueError("Durable evidence requires the real parent session ID")
        handle = result.metadata.get("result_id")
        if not isinstance(handle, str) or not handle:
            return None
        source_session = source_session_id if source_session_id is not None else parent_session_id
        if source_session != parent_session_id and source_session != child_namespace:
            raise PermissionError("Raw evidence source scope is not this parent or its host-owned child")
        if source_session != parent_session_id and not child_namespace.startswith(parent_session_id + "/child/"):
            raise PermissionError("Child evidence namespace is not owned by this parent")
        # get() is a direct in-memory lookup with no I/O; snapshot before yielding.
        stored = self.source_store.get(source_session, handle)
        if stored is None:
            raise RuntimeError("Tool preview references raw evidence that is already unavailable")
        metadata = {"created_at": stored.created_at.isoformat(), "source_session": source_session,
                    "content_type": "original_tool_result", "tool_name": stored.tool_name}
        await asyncio.to_thread(self._save, parent_session_id, handle, child_namespace, stored.tool_name, stored.content, metadata)
        return handle

    def _read(self, parent: str, handle: str, child: Optional[str]) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM pi_durable_evidence WHERE parent=? AND handle=?", (parent, handle)).fetchone()
            if row is None or (child is not None and row["child"] != child):
                raise PermissionError("Evidence handle is not available in this session")
            return dict(row)

    async def read(
        self, parent_session_id: str, handle: str, offset: int = 0, limit: int = 2800,
        keyword: str = "", *, child_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        row = await asyncio.to_thread(self._read, parent_session_id, handle, child_namespace)
        page = await asyncio.to_thread(_page, row["raw"], offset, limit, keyword)
        return {"result_id": handle, "tool_name": row["tool_name"], "metadata": json.loads(row["metadata"]), **page}

    async def read_archive(
        self, ledger: Any, parent_session_id: str, event_ids: Sequence[str], offset: int = 0, limit: int = 2800,
    ) -> Dict[str, Any]:
        if not event_ids or len(event_ids) > 20 or any(not isinstance(item, str) or not item for item in event_ids):
            raise ValueError("Provide between 1 and 20 archived event IDs")
        events = await asyncio.to_thread(ledger.archive_events, parent_session_id, event_ids)
        def render() -> Dict[str, Any]:
            text = json.dumps(
                [{"event_id": event.event_id, **public_archive_payload(event.payload)} for event in events],
                ensure_ascii=False, sort_keys=True,
            )
            return {"content_type": "archived_public_text_and_tool_evidence", **_page(text, offset, limit)}

        return await asyncio.to_thread(render)
