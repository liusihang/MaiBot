"""Batch new user evidence without spending a model request on every bot reply."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
import hashlib
import json
import sqlite3
import time

from json_repair import repair_json
from sqlmodel import select

from src.chat.message_receive.message import SessionMessage
from src.chat.utils.utils import is_bot_self
from src.common.database.database import ROOT_PATH, get_db_session
from src.common.database.database_model import Messages
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import global_config
from src.person_info.person_info import Person, get_person_id
from src.services.llm_service import LLMServiceClient
from src.services.memory_records import direct_self_evidence, fact_key, message_identity, user_text
from src.services.memory_service import memory_service

logger = get_logger("person_fact_writeback")
BATCH_INTERVAL_SECONDS = 600
BATCH_MAX_MESSAGES = 100
BATCH_MAX_CHARS = 24000


class PersonFactBatchJournal:
    """Persist message IDs, extracted results and the per-chat next attempt time."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS person_fact_jobs (
                session_id TEXT NOT NULL, message_id TEXT NOT NULL,
                queued_at REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                facts_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (session_id, message_id))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS person_fact_batches (
                session_id TEXT PRIMARY KEY, next_attempt_at REAL NOT NULL)""")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_person_fact_jobs_pending "
                "ON person_fact_jobs(session_id,status,queued_at)"
            )

    def enqueue(self, session_id: str, message_id: str, now: float) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO person_fact_batches VALUES (?,?)", (session_id, now + BATCH_INTERVAL_SECONDS)
            )
            connection.execute(
                "INSERT OR IGNORE INTO person_fact_jobs(session_id,message_id,queued_at) VALUES (?,?,?)",
                (session_id, message_id, now),
            )

    def due_sessions(self, now: float) -> List[str]:
        with closing(self._connect()) as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT b.session_id FROM person_fact_batches b WHERE b.next_attempt_at<=? "
                    "AND EXISTS (SELECT 1 FROM person_fact_jobs j "
                    "WHERE j.session_id=b.session_id AND j.status!='done') ORDER BY b.next_attempt_at",
                    (now,),
                )
            ]

    def claim(self, session_id: str, now: float) -> bool:
        # Reserve before calling the model, including failed/cancelled attempts and restarts.
        with closing(self._connect()) as connection, connection:
            result = connection.execute(
                "UPDATE person_fact_batches SET next_attempt_at=? WHERE session_id=? AND next_attempt_at<=?",
                (now + BATCH_INTERVAL_SECONDS, session_id, now),
            )
            return result.rowcount == 1

    def jobs(self, session_id: str, status: str) -> List[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM person_fact_jobs WHERE session_id=? AND status=? "
                    "ORDER BY queued_at,message_id LIMIT ?",
                    (session_id, status, BATCH_MAX_MESSAGES),
                )
            ]

    def extracted(self, session_id: str, message_ids: List[str], facts: List[Dict[str, Any]]) -> None:
        with closing(self._connect()) as connection, connection:
            for message_id in message_ids:
                own_facts = [item for item in facts if item["evidence_message_id"] == message_id]
                connection.execute(
                    "UPDATE person_fact_jobs SET status='extracted',facts_json=? WHERE session_id=? AND message_id=?",
                    (json.dumps(own_facts, ensure_ascii=False), session_id, message_id),
                )

    def complete(self, session_id: str, message_ids: List[str]) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executemany(
                "UPDATE person_fact_jobs SET status='done',facts_json='[]' WHERE session_id=? AND message_id=?",
                [(session_id, message_id) for message_id in message_ids],
            )


class PersonFactWritebackService:
    def __init__(self, journal_path: Optional[Path] = None) -> None:
        self._journal = PersonFactBatchJournal(journal_path or ROOT_PATH / "data/person_fact_writeback.sqlite3")
        self._worker_task: Optional[asyncio.Task] = None
        self._start_lock = asyncio.Lock()
        self._stopping = False
        self._extractor: Any | None = None

    async def start(self) -> None:
        async with self._start_lock:
            if self._worker_task is not None and not self._worker_task.done():
                return
            await asyncio.to_thread(self._journal.initialize)
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker_loop(), name="A_Memorix.person_fact_batches")
            logger.info(f"人物事实批处理已启动: interval={BATCH_INTERVAL_SECONDS}s max_messages={BATCH_MAX_MESSAGES}")

    async def shutdown(self) -> None:
        self._stopping = True
        worker, self._worker_task = self._worker_task, None
        if worker is not None:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _eligible(message: Any) -> bool:
        session_id, _, user_id = message_identity(message)
        return bool(
            session_id
            and user_id
            and message.message_id
            and message.platform
            and not is_bot_self(message.platform, user_id)
            and not message.is_command
            and user_text(message)
        )

    @staticmethod
    def _allowed(message: Any) -> bool:
        # WebUI and the bot import services on separate threads during startup.
        # Resolve the kernel's filter only when handling messages, after initialization.
        from src.A_memorix.core.runtime.services.chat_filter_service import MemoryChatFilterService

        session_id, group_id, user_id = message_identity(message)
        config = global_config.a_memorix.filter
        return MemoryChatFilterService._chat_filter_config_allows(
            {"enabled": config.enabled, "mode": config.mode, "chats": config.chats},
            stream_id=session_id,
            group_id=group_id,
            user_id=user_id,
        )

    async def enqueue(self, message: Any) -> None:
        if self._stopping or not global_config.a_memorix.integration.person_fact_writeback_enabled:
            return
        await asyncio.to_thread(self._enqueue, message)

    def _enqueue(self, message: Any) -> None:
        if self._eligible(message) and self._allowed(message):
            self._journal.enqueue(message_identity(message)[0], str(message.message_id), time.time())

    async def _worker_loop(self) -> None:
        while not self._stopping:
            try:
                await self.run_due()
            except Exception:
                logger.exception("人物事实批处理调度失败")
            await asyncio.sleep(5)

    async def run_due(self) -> None:
        if not global_config.a_memorix.integration.person_fact_writeback_enabled:
            return
        for session_id in await asyncio.to_thread(self._journal.due_sessions, time.time()):
            if not await asyncio.to_thread(self._journal.claim, session_id, time.time()):
                continue
            try:
                # If persistence failed after extraction, reuse the saved output without another model request.
                await self._persist_extracted(session_id)
                await self._extract_pending(session_id)
                await self._persist_extracted(session_id)
            except Exception:
                logger.exception(f"人物事实批处理失败，保留任务到下个窗口: session={session_id}")

    @staticmethod
    def _load_messages(session_id: str, message_ids: List[str]) -> Dict[str, SessionMessage]:
        with get_db_session(auto_commit=False) as session:
            rows = session.exec(
                select(Messages)
                .where(Messages.session_id == session_id, Messages.message_id.in_(message_ids))
                .order_by(Messages.id)
            ).all()
        return {row.message_id: SessionMessage.from_db_instance(row) for row in rows}

    async def _extract_pending(self, session_id: str) -> None:
        jobs = await asyncio.to_thread(self._journal.jobs, session_id, "pending")
        if not jobs:
            return
        sources = await asyncio.to_thread(self._load_messages, session_id, [j["message_id"] for j in jobs])
        selected: Dict[str, Any] = {}
        people: Dict[str, Dict[str, str]] = {}
        discarded: List[str] = []
        size = 0
        for job in jobs:
            message_id = job["message_id"]
            message = sources.get(message_id)
            if message is None or not await asyncio.to_thread(self._eligible, message) or not self._allowed(message):
                discarded.append(message_id)
                continue
            text = user_text(message)
            if selected and size + len(text) > BATCH_MAX_CHARS:
                break  # Keep overflow pending for a later window; never launch a second model call here.
            person = await asyncio.to_thread(
                Person, person_id=get_person_id(message.platform, message_identity(message)[2])
            )
            if not person.is_known:
                raise RuntimeError(f"用户身份尚未登记: message_id={message_id}")
            people[person.person_id] = {"person_id": person.person_id, "name": person.person_name or person.nickname}
            selected[message_id] = message
            size += len(text)
        if discarded:
            await asyncio.to_thread(self._journal.complete, session_id, discarded)
        if not selected:
            return
        facts = await self._extract_facts(session_id, selected, people)
        valid = []
        for item in facts:
            source = selected.get(str(item.get("evidence_message_id", "")))
            if source is None:
                continue
            person_id = get_person_id(source.platform, message_identity(source)[2])
            fact = item.get("fact")
            if item.get("person_id") != person_id or not isinstance(fact, str) or len(fact.strip()) < 4:
                continue
            if direct_self_evidence(item, [source], people[person_id]["name"]) is not None:
                valid.append({**item, "fact": fact.strip(), "evidence_message_id": str(source.message_id)})
        await asyncio.to_thread(self._journal.extracted, session_id, list(selected), valid)
        logger.info(
            f"人物事实批量抽取完成: session={session_id} messages={len(selected)} "
            f"people={len(people)} facts={len(valid)} interval={BATCH_INTERVAL_SECONDS}s"
        )

    async def _extract_facts(
        self, session_id: str, messages: Dict[str, Any], people: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        evidence = [
            {"message_id": mid, "person_id": get_person_id(m.platform, message_identity(m)[2]), "text": user_text(m)}
            for mid, m in messages.items()
        ]
        prompt = load_prompt(
            "person_fact_batch",
            people=json.dumps(list(people.values()), ensure_ascii=False),
            messages=json.dumps(evidence, ensure_ascii=False),
        )
        if self._extractor is None:
            self._extractor = LLMServiceClient(task_name="utils", request_type="A_Memorix.person_fact_writeback")
        result = await self._extractor.generate_response(prompt, session_id=session_id)
        payload = json.loads(repair_json(result.response))
        if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
            raise ValueError("人物事实批量抽取结果必须为 JSON 对象数组")
        return payload

    async def _persist_extracted(self, session_id: str) -> None:
        jobs = await asyncio.to_thread(self._journal.jobs, session_id, "extracted")
        sources = (
            await asyncio.to_thread(self._load_messages, session_id, [j["message_id"] for j in jobs]) if jobs else {}
        )
        existing: Dict[str, set] = {}
        for job in jobs:
            facts = json.loads(job["facts_json"])
            source = sources.get(job["message_id"])
            if facts and source is None:
                raise RuntimeError(f"人物事实原始证据不存在: {job['message_id']}")
            for item in facts:
                person_id = get_person_id(source.platform, message_identity(source)[2])
                if item["person_id"] != person_id or direct_self_evidence(item, [source]) is None:
                    raise ValueError("持久化人物事实与原始作者或证据不匹配")
                person = await asyncio.to_thread(Person, person_id=person_id)
                name = person.person_name or person.nickname
                if person_id not in existing:
                    result = await memory_service.fact_admin(
                        action="list", scope_type="person", scope_id=person_id, statuses=["active"], limit=200
                    )
                    if not result.get("success"):
                        raise RuntimeError("读取已有事实失败，保留已抽取结果")
                    existing[person_id] = {fact_key(f["value_text"], name) for f in result.get("items", [])}
                key = fact_key(item["fact"], name)
                if key in existing[person_id]:
                    continue
                _, group_id, user_id = message_identity(source)
                digest = hashlib.md5(f"{person_id}|{session_id}|{item['fact']}".encode()).hexdigest()
                section = item.get("profile_section", "stable_facts")
                if section not in {"stable_facts", "identity_settings", "interaction_preferences"}:
                    section = "stable_facts"
                result = await memory_service.ingest_text(
                    external_id=f"person_fact:{person_id}:{digest}",
                    source_type="person_fact",
                    text=item["fact"],
                    chat_id=session_id,
                    person_ids=[person_id],
                    participants=[name],
                    tags=["person_fact"],
                    metadata={
                        "platform": source.platform,
                        "person_id": person_id,
                        "person_name": name,
                        "writeback_source": "memory_flow_service",
                        "evidence_source": "direct_self_statement",
                        "evidence_message_ids": [job["message_id"]],
                        "fact_claim": {"authority": "direct_user", "stability": "stable", "profile_section": section},
                    },
                    respect_filter=True,
                    group_id=group_id,
                    user_id=user_id,
                )
                if (
                    not result.success
                    or result.detail == "chat_filtered"
                    or not (result.stored_ids or result.skipped_ids)
                ):
                    raise RuntimeError(f"人物事实未获保存回执: {result.detail}")
                existing[person_id].add(key)
            await asyncio.to_thread(self._journal.complete, session_id, [job["message_id"]])
