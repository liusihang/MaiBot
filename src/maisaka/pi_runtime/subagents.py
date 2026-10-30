"""Isolated Pi delegation and durable, parent-scoped access to child evidence.

The injected runner must execute one real Pi session, not another host model loop.
This module owns persistence and authorization, not model or tool orchestration.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Mapping, Sequence

import asyncio
import json
import sqlite3
import uuid


READ_ONLY_TOOL_NAMES = frozenset(
    {
        "query_memory",
        "query_person_profile",
        "query_image_memory",
        "read_tool_result",
        "view_forward_message",
        "web_search",
        "web_fetch",
        "gpt_web_search",
        "gpt_read_search_result",
    }
)
MAX_CONCLUSION_CHARS = 4000
MAX_READ_CHARS = 12000
SUBAGENT_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "subagent",
            "description": "Delegate multi-step research or complex actions to an independent Pi thread with the same tools. Supply a self-contained task, relevant message IDs, and the desired deliverable. wait=true (default) returns its conclusion and evidence handle; wait=false returns a handle immediately and delivers completion to this chat later. Ordinary chat and one-step tasks can be handled directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "minLength": 1},
                    "wait": {
                        "type": "boolean",
                        "default": True,
                        "description": "Wait for the result, or continue chatting while the child works.",
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_subagent",
            "description": "Read a delegated task's status, conclusion and original tool evidence in this chat, including after restart. Set wait=true to await a running task instead of polling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": {"type": "string"},
                    "wait": {"type": "boolean", "default": False},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_READ_CHARS},
                },
                "required": ["handle"],
                "additionalProperties": False,
            },
        },
    },
]
EventSink = Callable[[Mapping[str, Any]], Awaitable[None]]
ChildRunner = Callable[[str, str, str, Sequence[Dict[str, Any]], EventSink], Awaitable[Dict[str, Any]]]


def _tool_name(definition: Mapping[str, Any]) -> str:
    function = definition.get("function", definition)
    return str(function.get("name", "")) if isinstance(function, dict) else ""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class PiSubagentService:
    """Persistence has no completed-task eviction; a handle is not a temporary cache ID."""

    def __init__(self, db_path: str, run_child: ChildRunner, on_complete: EventSink | None = None) -> None:
        self.db_path = str(db_path)
        self.run_child = run_child
        self.on_complete = on_complete
        self._tasks: Dict[str, asyncio.Task] = {}
        self._closing = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS pi_children (
                handle TEXT PRIMARY KEY, parent TEXT NOT NULL, child TEXT UNIQUE NOT NULL,
                task TEXT NOT NULL, grants TEXT NOT NULL, status TEXT NOT NULL,
                result TEXT, error TEXT,
                background INTEGER NOT NULL DEFAULT 0, notification_queued INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS pi_child_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                handle TEXT NOT NULL REFERENCES pi_children(handle), payload TEXT NOT NULL
            );
        """)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(pi_children)")}
        for column in ("background", "notification_queued"):
            if column not in columns:
                connection.execute(f"ALTER TABLE pi_children ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def allowed_tool_names(tools: Sequence[Mapping[str, Any]]) -> List[str]:
        """The child receives the same frozen tool set as its group parent."""
        return sorted({_tool_name(tool) for tool in tools})

    @staticmethod
    def assert_tool_allowed(name: str, granted_names: Sequence[str]) -> None:
        if name not in granted_names:
            raise PermissionError(f"Tool is not bound to this child session: {name}")

    def _create(
        self, handle: str, parent: str, child: str, task: str, grants: List[str], background: bool = False
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO pi_children(handle,parent,child,task,grants,status,background) VALUES(?,?,?,?,?,?,?)",
                (handle, parent, child, task, _json(grants), "running", int(background)),
            )

    def _append(self, handle: str, event: Mapping[str, Any]) -> None:
        # A child evidence cache is business data, never a second native model
        # transcript. Project at write time as well as at public read time.
        projected = self._public_event(event)
        if projected is None:
            return
        with self._connect() as connection:
            connection.execute("INSERT INTO pi_child_events(handle,payload) VALUES(?,?)", (handle, _json(projected)))

    def _finish(self, handle: str, status: str, result: Any, error: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE pi_children SET status=?,result=?,error=? WHERE handle=?",
                (status, _json(result), error, handle),
            )

    async def delegate(
        self,
        parent_session_id: str,
        task: str,
        tools: Sequence[Dict[str, Any]],
        *,
        wait: bool = True,
    ) -> Dict[str, Any]:
        if not parent_session_id.strip() or not task.strip():
            raise ValueError("A real parent session ID and a non-empty task are required")
        handle = "pi-child-" + uuid.uuid4().hex
        # This is a child execution ID, never a replacement chat ownership ID.
        child_id = parent_session_id + "/child/" + uuid.uuid4().hex
        grants = self.allowed_tool_names(tools)
        await asyncio.to_thread(self._create, handle, parent_session_id, child_id, task, grants, not wait)
        execution = asyncio.create_task(self._execute(handle, child_id, parent_session_id, task, list(tools), not wait))
        self._tasks[handle] = execution

        def completed(task: asyncio.Task) -> None:
            self._tasks.pop(handle, None)
            if not task.cancelled():
                task.exception()

        execution.add_done_callback(completed)
        if wait:
            return await asyncio.shield(execution)
        return {"handle": handle, "status": "running", "evidence_handle": handle}

    async def _execute(
        self, handle: str, child_id: str, parent: str, task: str, tools: List[Dict[str, Any]], background: bool
    ) -> Dict[str, Any]:

        async def on_event(event: Mapping[str, Any]) -> None:
            await asyncio.to_thread(self._append, handle, event)

        try:
            raw_result = await self.run_child(child_id, parent, task, tools, on_event)
            result = {
                "conclusion": self._conclusion(raw_result),
                "sources": raw_result.get("sources", []),
                "uncertainty": raw_result.get("uncertainty"),
                "archive_handles": raw_result.get("archiveHandles", []),
            }
            await asyncio.to_thread(self._finish, handle, "completed", result)
        except asyncio.CancelledError:
            # Shield the durable terminal marker; no fabricated successful conclusion.
            await asyncio.shield(asyncio.to_thread(self._finish, handle, "cancelled", None))
            raise
        except Exception as exc:
            await asyncio.to_thread(self._finish, handle, "failed", None, type(exc).__name__)
        finally:
            if background and not self._closing:
                await asyncio.shield(self._notify(parent, handle))
        return await asyncio.to_thread(self._summary, parent, handle)

    def _summary(self, parent: str, handle: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pi_children WHERE handle=? AND parent=?", (handle, parent)
            ).fetchone()
        if row is None:
            raise PermissionError("Child evidence handle is not available in this parent session")
        result = (json.loads(row["result"]) if row["result"] else None) or {}
        conclusion = self._conclusion(result)
        sources, uncertainty = result.get("sources"), result.get("uncertainty")
        return {
            "handle": handle,
            "status": row["status"],
            "error": row["error"],
            "conclusion": conclusion[:MAX_CONCLUSION_CHARS],
            "conclusion_truncated": len(conclusion) > MAX_CONCLUSION_CHARS,
            # Never manufacture source URLs from arbitrary assistant prose.
            "sources": [source for source in sources[:20] if len(_json(source)) <= 1000]
            if isinstance(sources, list)
            else [],
            "uncertainty": uncertainty[:1000]
            if isinstance(uncertainty, str)
            else "No separately structured uncertainty statement was returned; inspect the conclusion and original evidence.",
            "evidence_handle": handle,
        }

    async def _notify(self, parent: str, handle: str) -> None:
        if self.on_complete is None:
            return
        await self.on_complete(await asyncio.to_thread(self._summary, parent, handle))
        await asyncio.to_thread(self._mark_notified, parent, handle)

    def _mark_notified(self, parent: str, handle: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE pi_children SET notification_queued=1 WHERE parent=? AND handle=?", (parent, handle)
            )

    def _recover(self, parent: str) -> List[str]:
        with self._connect() as connection:
            for row in connection.execute(
                "SELECT handle FROM pi_children WHERE parent=? AND status='running'", (parent,)
            ).fetchall():
                if row["handle"] not in self._tasks:
                    # A stopped process cannot still own this task. Never replay actions automatically.
                    connection.execute(
                        "UPDATE pi_children SET status='interrupted',error='Process stopped before completion' WHERE handle=?",
                        (row["handle"],),
                    )
            return [
                row[0]
                for row in connection.execute(
                    "SELECT handle FROM pi_children WHERE parent=? AND background=1 AND status!='running' AND notification_queued=0",
                    (parent,),
                ).fetchall()
            ]

    async def recover(self, parent: str) -> None:
        if Path(self.db_path).exists():
            for handle in await asyncio.to_thread(self._recover, parent):
                await self._notify(parent, handle)

    async def close(self) -> None:
        self._closing = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _conclusion(result: Mapping[str, Any]) -> str:
        if isinstance(result.get("conclusion"), str):
            return result["conclusion"]
        messages = result.get("messages", [])
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", [])
            if isinstance(content, str):
                return content
            if any(part.get("type") == "toolCall" for part in content):
                continue
            return "\n".join(part["text"] for part in content if part.get("type") == "text")
        return ""

    @staticmethod
    def _public_event(event: Mapping[str, Any]) -> Dict[str, Any] | None:
        """Project evidence, not provider replay or private chain-of-thought."""
        event = event.get("event", event)
        if event.get("type") in {"tool_evidence", "public_conclusion"}:
            return {key: event.get(key, "") for key in ("type", "tool_name", "call_id", "text")}
        if event.get("type") != "message_end":
            return None
        message = event.get("message", {})
        role = message.get("role")
        if role not in {"toolResult", "assistant"}:
            return None
        content = message.get("content", [])
        if isinstance(content, str):
            text = content
        else:
            if role == "assistant" and any(part.get("type") == "toolCall" for part in content):
                return None
            text = "\n".join(part["text"] for part in content if part.get("type") == "text")
        if not text:
            return None
        return {
            "type": "tool_evidence" if role == "toolResult" else "public_conclusion",
            "tool_name": message.get("toolName", ""),
            "call_id": message.get("toolCallId", ""),
            "text": text,
        }

    def _read(self, parent: str, handle: str, offset: int, limit: int) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pi_children WHERE handle=? AND parent=?",
                (handle, parent),
            ).fetchone()
            if row is None:
                # Same error for an unknown or a different parent's handle.
                raise PermissionError("Child evidence handle is not available in this parent session")
            events = [
                json.loads(event[0])
                for event in connection.execute(
                    "SELECT payload FROM pi_child_events WHERE handle=? ORDER BY seq",
                    (handle,),
                )
            ]
        result = json.loads(row["result"]) if row["result"] else None
        public_result = (
            None
            if not isinstance(result, dict)
            else {
                "conclusion": self._conclusion(result),
                "sources": result.get("sources", []),
                "uncertainty": result.get("uncertainty", ""),
            }
        )
        transcript = _json(
            {
                "content_type": "child_public_conclusion_and_tool_evidence",
                "task": row["task"],
                "status": row["status"],
                "events": [projected for event in events if (projected := self._public_event(event)) is not None],
                "result": public_result,
                "error": row["error"],
            }
        )
        end = min(offset + limit, len(transcript))
        return {
            "handle": handle,
            "status": row["status"],
            "offset": offset,
            "text": transcript[offset:end],
            "next_offset": end if end < len(transcript) else None,
            "total_chars": len(transcript),
        }

    async def read(
        self,
        parent_session_id: str,
        handle: str,
        offset: int = 0,
        limit: int = MAX_READ_CHARS,
        *,
        wait: bool = False,
    ) -> Dict[str, Any]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= MAX_READ_CHARS:
            raise ValueError("Invalid child transcript pagination")
        # Ownership is checked before joining, even for an in-process task.
        summary = await asyncio.to_thread(self._summary, parent_session_id, handle)
        if wait and handle in self._tasks:
            await asyncio.shield(self._tasks[handle])
            summary = await asyncio.to_thread(self._summary, parent_session_id, handle)
        return {**summary, **await asyncio.to_thread(self._read, parent_session_id, handle, offset, limit)}
