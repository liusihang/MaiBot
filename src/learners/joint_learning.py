"""One evidence-backed extraction for expressions and jargon per learning window."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
import json
import time

from sqlmodel import select

from src.chat.message_receive.message import SessionMessage
from src.chat.utils.utils import is_bot_self
from src.common.database.database import ROOT_PATH, get_db_session
from src.common.database.database_model import Messages
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import global_config
from src.services.llm_service import LLMServiceClient

from .expression_learner import expression_learning_batch_gate
from .expression_provenance import ExpressionProvenance, expression_identity, text_digest
from .expression_style_utils import normalize_expression_style_for_learning
from .jargon_learner import JargonLearner
from .joint_learning_journal import JointLearningJournal
from .learning_text import extract_learning_text

logger = get_logger("joint_learning")


def parse_extraction(text: str, source_count: int) -> Dict[str, List[Dict[str, str]]]:
    """Fail closed on malformed output; candidate IDs always refer to this batch."""
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {"expressions", "jargon"}:
        raise ValueError("Joint extraction requires expressions and jargon arrays")
    result: Dict[str, List[Dict[str, str]]] = {"expressions": [], "jargon": []}
    for kind, keys, limit in [("expressions", {"situation", "style", "id"}, 20), ("jargon", {"content", "id"}, 30)]:
        values = obj[kind]
        if not isinstance(values, list) or len(values) > limit:
            raise ValueError("Invalid joint extraction candidate count")
        for value in values:
            if not isinstance(value, dict) or set(value) != keys or any(not isinstance(v, str) for v in value.values()):
                raise ValueError("Invalid joint extraction candidate schema")
            if not value["id"].isdigit() or not 1 <= int(value["id"]) <= source_count:
                raise ValueError("Joint extraction source ID is outside the batch")
            result[kind].append({key: item.strip() for key, item in value.items()})
    return result


def parse_review(text: str, candidate_count: int) -> List[int]:
    result = json.loads(text)
    if not isinstance(result, list) or len(result) != candidate_count:
        raise ValueError("Review must cover every candidate exactly once")
    approved, seen = [], set()
    for item in result:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "suitable", "reason"}
            or type(item["id"]) is not int
            or type(item["suitable"]) is not bool
            or not isinstance(item["reason"], str)
            or not 0 <= item["id"] < candidate_count
            or item["id"] in seen
        ):
            raise ValueError("Invalid evidence review result")
        seen.add(item["id"])
        if item["suitable"]:
            approved.append(item["id"])
    return approved


class JointLearningCoordinator:
    """Own a per-runtime timer; persisted receipts survive cancellation and restart."""

    def __init__(self, runtime: Any, path: Optional[Path] = None) -> None:
        self.runtime = runtime
        self.journal = JointLearningJournal(path or Path(ROOT_PATH) / "data" / "joint_learning.sqlite3")
        self.task: Optional[asyncio.Task] = None
        self.wakeup = asyncio.Event()
        self.initialized = False
        self.client: Optional[LLMServiceClient] = None
        self.review_client: Optional[LLMServiceClient] = None

    async def start(self) -> None:
        if not self.initialized:
            await asyncio.to_thread(self.journal.initialize)
            self.initialized = True
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run(), name=f"joint-learning:{self.runtime.session_id}")

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def collect(self, context_messages: Any) -> List[str]:
        if not (self.runtime._enable_expression_learning or self.runtime._enable_jargon_learning):
            return []
        await self.start()
        ids = []
        for context in context_messages:
            message = context.original_message
            if message.session_id != self.runtime.session_id:
                raise ValueError("Joint learning source belongs to another session")
            if (
                message.is_notify
                or is_bot_self(message.platform, message.message_info.user_info.user_id)
                or not extract_learning_text(message.raw_message).strip()
            ):
                continue
            ids.append(message.message_id)
        if ids:
            added = await asyncio.to_thread(self.journal.enqueue, self.runtime.session_id, ids, time.time())
            self.wakeup.set()
            return added
        return []

    async def run(self) -> None:
        while True:
            try:
                if self.runtime._enable_expression_learning or self.runtime._enable_jargon_learning:
                    batch = await asyncio.to_thread(self.journal.next_batch, self.runtime.session_id, time.time())
                    if batch:
                        acquired = await expression_learning_batch_gate.acquire(self.runtime.session_id)
                        if acquired.acquired:
                            try:
                                await self.process(batch)
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                terminal = await asyncio.to_thread(self.journal.fail, batch["id"], time.time())
                                logger.exception(
                                    "Joint learning failed; checkpoint retained, terminal=%s batch=%s",
                                    terminal,
                                    batch["id"],
                                )
                            finally:
                                await expression_learning_batch_gate.release(self.runtime.session_id)
                            continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Joint learning scheduler failed; durable queue retained")
            self.wakeup.clear()
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

    @staticmethod
    def load_sources(session_id: str, ids: List[str]) -> List[SessionMessage]:
        with get_db_session(auto_commit=False) as db:
            rows = db.exec(
                select(Messages).where(Messages.session_id == session_id, Messages.message_id.in_(ids))
            ).all()
        by_id = {row.message_id: SessionMessage.from_db_instance(row) for row in rows}
        if any(mid not in by_id for mid in ids):
            raise ValueError("Learning evidence has disappeared; refusing to invent sources")
        return [by_id[mid] for mid in ids]

    @staticmethod
    def review_window(session_id: str, message_id: str) -> Dict[str, Any]:
        """Read twenty actual neighbours, not the extraction batch's artificial edges."""
        with get_db_session(auto_commit=False) as db:
            source = db.exec(
                select(Messages).where(Messages.session_id == session_id, Messages.message_id == message_id)
            ).first()
            if source is None:
                raise ValueError("Expression review source is missing")
            before = db.exec(
                select(Messages)
                .where(Messages.session_id == session_id, Messages.id < source.id)
                .order_by(Messages.id.desc())
                .limit(10)
            ).all()
            after = db.exec(
                select(Messages)
                .where(Messages.session_id == session_id, Messages.id > source.id)
                .order_by(Messages.id)
                .limit(10)
            ).all()
        rows = [*reversed(before), source, *after]
        messages = []
        for row in rows:
            message = SessionMessage.from_db_instance(row)
            messages.append(
                {
                    "message_id": row.message_id,
                    "sender_id": row.user_id,
                    "sender": row.user_cardname or row.user_nickname,
                    "time": row.timestamp.isoformat(),
                    "is_source": row.id == source.id,
                    "text": extract_learning_text(message.raw_message),
                    "reply_to_message_id": row.reply_to,
                }
            )
        return {
            "before_count": len(before),
            "after_count": len(after),
            "complete": len(before) == 10 and len(after) == 10,
            "messages": messages,
        }

    async def process(self, batch: Dict[str, Any]) -> None:
        runtime, batch_id = self.runtime, batch["id"]
        messages = await asyncio.to_thread(self.load_sources, runtime.session_id, json.loads(batch["source_ids"]))
        if batch["extraction"] is None:
            source_data = [
                {
                    "id": str(i),
                    "sender_id": m.message_info.user_info.user_id,
                    "sender": m.message_info.user_info.user_nickname,
                    "text": extract_learning_text(m.raw_message),
                }
                for i, m in enumerate(messages, 1)
            ]
            if self.client is None:
                self.client = LLMServiceClient(task_name="learner", request_type="learning.joint_extraction")
            response = await self.client.generate_response(
                load_prompt(
                    "learn_joint",
                    messages=json.dumps(source_data, ensure_ascii=False),
                    expression_enabled=str(runtime._enable_expression_learning).lower(),
                    jargon_enabled=str(runtime._enable_jargon_learning).lower(),
                ),
                session_id=runtime.session_id,
            )
            extracted = parse_extraction(response.response, len(messages))
            await asyncio.to_thread(self.journal.save, batch_id, extraction=json.dumps(extracted, ensure_ascii=False))
        else:
            extracted = parse_extraction(batch["extraction"], len(messages))
        if not batch["expressions_done"]:
            await self.process_expressions(batch, messages, extracted["expressions"])
            await asyncio.to_thread(self.journal.save, batch_id, expressions_done=1)
        if not batch["jargon_done"]:
            if runtime._enable_jargon_learning:
                sources = [JargonLearner._build_source_item_from_session_message(m) for m in messages]
                jargon = [
                    item
                    for item in extracted["jargon"]
                    if item["content"]
                    and item["content"] in extract_learning_text(messages[int(item["id"]) - 1].raw_message)
                ]
                await runtime._jargon_learner._run_learning_batch(
                    sources,
                    learning_session_id=runtime.session_id,
                    jargon_miner=runtime._jargon_miner,
                    learned_contents=[],
                    extracted_response=json.dumps(jargon, ensure_ascii=False),
                )
            await asyncio.to_thread(self.journal.save, batch_id, jargon_done=1)
        await asyncio.to_thread(self.journal.save, batch_id, done=1)

    async def process_expressions(self, batch: Dict[str, Any], messages: Any, candidates: Any) -> None:
        runtime = self.runtime
        if not runtime._enable_expression_learning:
            return
        learner = runtime._expression_learner
        if batch["review"] is not None:
            accepted = json.loads(batch["review"])
        else:
            filtered, seen = [], set()
            provenance = await asyncio.to_thread(ExpressionProvenance(self.journal.path).load, runtime.session_id)
            prepared = learner._prepare_learning_messages(messages)
            for candidate in candidates:
                style = normalize_expression_style_for_learning(candidate["style"])
                key = (candidate["situation"], style)
                if key in seen or not learner._filter_expressions([(*key, candidate["id"])], prepared):
                    continue
                seen.add(key)
                existing = await asyncio.to_thread(
                    learner._find_similar_expression, *key, session_id=runtime.session_id, strict_errors=True
                )
                if existing:
                    expression = existing[0]
                    saved = provenance.get(expression.item_id, {})
                    identity = expression_identity(
                        expression.situation, expression.style, expression.create_time.isoformat()
                    )
                    if saved.get("identity") == identity and saved.get("sources"):
                        continue
                    # An old expression can acquire a real example when observed
                    # again. It still passes evidence review and reuses its ID.
                    if (expression.situation, expression.style) != key:
                        continue
                source = messages[int(candidate["id"]) - 1]
                filtered.append(
                    {
                        **candidate,
                        "style": style,
                        "source_ref": {
                            "message_id": source.message_id,
                            "platform": source.platform,
                            "text_hash": text_digest(extract_learning_text(source.raw_message)),
                            "example": extract_learning_text(source.raw_message)[:600],
                        },
                    }
                )
            accepted = filtered
            if filtered and global_config.expression.expression_self_reflect:
                evidence = [
                    {
                        "id": i,
                        "situation": item["situation"],
                        "style": item["style"],
                        "context": await asyncio.to_thread(
                            self.review_window, runtime.session_id, messages[int(item["id"]) - 1].message_id
                        ),
                    }
                    for i, item in enumerate(filtered)
                ]
                # Share overlapping neighbour messages once across the whole review request.
                shared_messages = {}
                for item in evidence:
                    window = item["context"]
                    rows = window.pop("messages")
                    window["message_ids"] = [row["message_id"] for row in rows]
                    for row in rows:
                        # source-ness is candidate-specific, not a global message property.
                        if row["is_source"]:
                            window["source_message_id"] = row["message_id"]
                        shared_messages[row["message_id"]] = {k: v for k, v in row.items() if k != "is_source"}
                if self.review_client is None:
                    self.review_client = LLMServiceClient(task_name="learner", request_type="expression.batch_review")
                result = await self.review_client.generate_response(
                    load_prompt(
                        "expression_batch_review",
                        candidates=json.dumps(
                            {"candidates": evidence, "messages": shared_messages}, ensure_ascii=False
                        ),
                    ),
                    session_id=runtime.session_id,
                )
                approved = parse_review(result.response, len(filtered))
                accepted = [filtered[i] for i in approved]
                for i in approved:
                    # Keep compact, attributed context references, not another copy of chat text.
                    window_ids = evidence[i]["context"]["message_ids"]
                    source_id = evidence[i]["context"]["source_message_id"]
                    position = window_ids.index(source_id)
                    filtered[i]["context_refs"] = [
                        {"message_id": mid, "text_hash": text_digest(shared_messages[mid]["text"])}
                        for mid in window_ids[max(0, position - 1) : position + 2]
                        if mid != source_id
                    ]
            # Checkpoint approved source-bound candidates before any database write.
            await asyncio.to_thread(self.journal.save, batch["id"], review=json.dumps(accepted, ensure_ascii=False))

        async def preserve_source(expression: Any) -> None:
            matched = [
                item
                for item in accepted
                if (item["situation"], item["style"]) == (expression.situation, expression.style)
            ]
            refs = []
            for item in matched:
                if "source_ref" in item:
                    refs.append({**item["source_ref"], "context_refs": item.get("context_refs", [])})
            # Old checkpoints did not retain a reviewed text fingerprint: do not retrofit evidence.
            if not refs:
                return
            if expression.item_id is None:
                raise ValueError("Expression receipt lacks a persisted record ID")
            identity = expression_identity(expression.situation, expression.style, expression.create_time.isoformat())
            await asyncio.to_thread(
                ExpressionProvenance(self.journal.path).save,
                expression.item_id,
                identity,
                runtime.session_id,
                refs,
            )

        # Preserve extraction and write hooks; evidence review already happened once.
        await learner._run_learning_batch(
            messages,
            learning_session_id=runtime.session_id,
            learned_contents=[],
            extracted_response=json.dumps(accepted, ensure_ascii=False),
            already_reviewed=True,
            on_written=preserve_source,
        )
