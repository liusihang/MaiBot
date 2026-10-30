from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional

import asyncio
import json
import time

from src.common.logger import get_logger
from src.common.database.database import ROOT_PATH
from src.common.message_repository import count_messages, find_messages
from src.common.data_models.message_component_data_model import TextComponent
from src.config.config import global_config
from src.services import memory_service as memory_service_module
from src.services.memory_service import memory_service
from src.A_memorix.core.image.component_paths import build_chat_external_ref, iter_message_image_components

from .image_writeback_journal import ImageWritebackJournal
from .memory_records import message_identity
from .person_fact_writeback import PersonFactWritebackService

logger = get_logger("memory_flow_service")


@dataclass
class ChatSummaryWritebackState:
    last_trigger_message_count: int = 0
    last_trigger_time: float = 0.0


class ChatSummaryWritebackService:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
        self._worker_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._states: dict[str, ChatSummaryWritebackState] = {}
        self._recovery_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._stopping = False
        self._worker_task = asyncio.create_task(self._worker_loop(), name="memory_chat_summary_writeback")
        self._recovery_task = asyncio.create_task(self._recover_recent(), name="memory_summary_recovery")

    async def shutdown(self) -> None:
        self._stopping = True
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass
            self._recovery_task = None
        worker = self._worker_task
        self._worker_task = None
        if worker is None:
            return
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(f"关闭聊天摘要写回 worker 失败: {exc}")

    async def enqueue(self, message: Any, *, flush: bool = False) -> None:
        if not bool(global_config.a_memorix.integration.chat_summary_writeback_enabled):
            return
        if self._stopping:
            return
        try:
            self._queue.put_nowait((message, flush))
        except asyncio.QueueFull:
            logger.warning("聊天摘要写回队列已满，跳过本次触发")

    async def _worker_loop(self) -> None:
        try:
            while not self._stopping:
                message, flush = await self._queue.get()
                try:
                    await self._handle_message(message, flush=flush)
                except Exception as exc:
                    logger.warning(f"聊天摘要写回处理失败: {exc}", exc_info=True)
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            raise

    async def _recover_recent(self) -> None:
        # Resume today's already recorded messages in the live process/kernel.
        await asyncio.sleep(5)
        cfg = global_config.a_memorix.filter
        groups = [str(x).split(":", 1)[1] for x in cfg.chats if str(x).startswith("group:")]
        for group_id in groups:
            messages = await asyncio.to_thread(find_messages, group_id=group_id, limit=1, limit_mode="latest")
            if messages and not self._stopping:
                await self.enqueue(messages[0], flush=True)

    async def _handle_message(self, message: Any, *, flush: bool = False) -> None:
        session_id, group_id, user_id = message_identity(message)
        if not session_id:
            return
        message_time = self._extract_message_timestamp(message)
        total = await asyncio.to_thread(
            self._count_messages_until_trigger, session_id=session_id, message_time=message_time,
        )
        if total <= 0:
            return
        state = self._states.get(session_id)
        if state is None:
            restored = await self._load_last_trigger_message_count(session_id=session_id, total_message_count=total)
            if restored == 0:
                midnight = datetime.now(ZoneInfo("Asia/Shanghai")).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
                restored = await asyncio.to_thread(count_messages, session_id=session_id, before_time=midnight)
            state = ChatSummaryWritebackState(last_trigger_message_count=restored)
            self._states[session_id] = state
        pending = total - state.last_trigger_message_count
        threshold = self._message_threshold()
        if pending < threshold and not flush:
            return
        if pending <= 0:
            return
        history = await asyncio.to_thread(
            find_messages, session_id=session_id, end_time=message_time, limit=total, limit_mode="earliest",
        )
        while total > state.last_trigger_message_count:
            pending = total - state.last_trigger_message_count
            if pending < threshold and not flush:
                break
            size = min(self._context_length(), pending)
            end = state.last_trigger_message_count + size
            # Keep a timestamp boundary intact, so equal-time messages cannot be skipped.
            while end < len(history) and self._extract_message_timestamp(history[end]) == self._extract_message_timestamp(history[end-1]):
                end += 1
            batch = history[state.last_trigger_message_count:end]
            if not batch:
                break
            result = await memory_service.ingest_summary(
                external_id=f"chat_auto_summary:v2:{session_id}:{end}", chat_id=session_id,
                text="", participants=[], time_end=self._extract_message_timestamp(batch[-1]),
                metadata={"generate_from_chat": True, "writeback_version": 2, "context_length": len(batch),
                    "include_personality": False, "writeback_source": "memory_flow_service",
                    "trigger": "recovery" if flush else "message_threshold",
                    "previous_trigger_message_count": state.last_trigger_message_count,
                    "pending_message_count": pending, "trigger_message_count": end, "summary_review_count": 2},
                respect_filter=True, user_id=user_id, group_id=group_id,
            )
            if not result.success or result.detail == "chat_filtered":
                logger.warning(f"聊天摘要未保存: session_id={session_id} group_id={group_id} detail={result.detail}")
                return
            # A persisted no-new-information checkpoint is legitimate; a filter skip is not.
            if not result.stored_ids and not result.skipped_ids:
                logger.warning(f"聊天摘要缺少保存回执: session_id={session_id}")
                return
            state.last_trigger_message_count = end
            state.last_trigger_time = time.time()
            logger.info(f"聊天摘要自动写回成功: session_id={session_id} total_messages={end} context_length={len(batch)} detail={result.detail}")

    async def _load_last_trigger_message_count(self, *, session_id: str, total_message_count: int) -> int:
        """从已落库的聊天摘要恢复触发游标，避免服务重启后重复摘要。"""
        try:
            runtime_manager = getattr(memory_service_module, "a_memorix_host_service", None)
            ensure_kernel = getattr(runtime_manager, "_ensure_kernel", None)
            if not callable(ensure_kernel):
                return 0

            kernel = await ensure_kernel()
            metadata_store = getattr(kernel, "metadata_store", None)
            if metadata_store is None:
                return 0

            checkpoint_rows, paragraphs = await asyncio.to_thread(
                lambda: (
                    metadata_store.query(
                        "SELECT COALESCE(MAX(trigger_message_count),0) AS count "
                        "FROM chat_summary_checkpoints WHERE chat_id=? AND external_id LIKE ?",
                        (session_id, f"chat_auto_summary:v2:{session_id}:%"),
                    ),
                    metadata_store.get_paragraphs_by_source(f"chat_summary:{session_id}"),
                )
            )
            checkpoint_count = int(checkpoint_rows[0]["count"]) if checkpoint_rows else 0
            paragraphs = [
                paragraph for paragraph in paragraphs
                if self._paragraph_metadata(paragraph).get("writeback_version") == 2
            ]
            if not paragraphs:
                return min(total_message_count, checkpoint_count)

            # 正文去重会复用较早的段落，created_at 不能代表最后完成的窗口。
            trigger_message_count = max(
                (
                    self._coerce_positive_int(metadata.get("trigger_message_count"))
                    for paragraph in paragraphs
                    if (metadata := self._paragraph_metadata(paragraph)).get("chat_id", session_id) == session_id
                ),
                default=0,
            )
            if max(trigger_message_count, checkpoint_count) > 0:
                return min(total_message_count, max(trigger_message_count, checkpoint_count))

            # 兼容旧摘要数据：没有触发计数时，只能退化为对齐当前计数，
            # 至少避免重启后立刻重复写入一条相近摘要。
            return total_message_count
        except Exception as exc:
            logger.debug(f"恢复聊天摘要写回游标失败: session_id={session_id} error={exc}")
            return 0

    @staticmethod
    def _paragraph_created_at(paragraph: dict[str, Any]) -> float:
        try:
            return float(paragraph.get("created_at") or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _paragraph_metadata(paragraph: dict[str, Any]) -> dict[str, Any]:
        metadata = paragraph.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, (bytes, bytearray)):
            try:
                metadata = metadata.decode("utf-8")
            except Exception:
                return {}
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _coerce_positive_int(value: Any) -> int:
        try:
            number = int(value or 0)
        except Exception:
            return 0
        return max(0, number)

    @staticmethod
    def _resolve_session_id(message: Any) -> str:
        return str(
            getattr(message, "session_id", "") or getattr(getattr(message, "session", None), "session_id", "") or ""
        ).strip()

    @staticmethod
    def _extract_session_user_id(message: Any) -> str:
        return message_identity(message)[2]

    @staticmethod
    def _extract_session_group_id(message: Any) -> str:
        return message_identity(message)[1]

    @staticmethod
    def _extract_message_timestamp(message: Any) -> float | None:
        raw_timestamp = getattr(message, "timestamp", None)
        if isinstance(raw_timestamp, datetime):
            return raw_timestamp.timestamp()
        if hasattr(raw_timestamp, "timestamp") and callable(raw_timestamp.timestamp):
            try:
                return float(raw_timestamp.timestamp())
            except Exception:
                return None
        if isinstance(raw_timestamp, (int, float)):
            return float(raw_timestamp)
        return None

    @staticmethod
    def _message_threshold() -> int:
        return max(1, int(global_config.a_memorix.integration.chat_summary_writeback_message_threshold))

    @staticmethod
    def _context_length() -> int:
        return max(1, int(global_config.a_memorix.integration.chat_summary_writeback_context_length))

    @staticmethod
    def _count_messages_until_trigger(*, session_id: str, message_time: float | None) -> int:
        if message_time is None:
            return count_messages(session_id=session_id)
        return count_messages(session_id=session_id, end_time=message_time)

    @staticmethod
    def _effective_context_length(*, configured_context_length: int, pending_message_count: int) -> int:
        """摘要只覆盖本轮新增消息，避免重叠窗口反复消耗 token。"""
        configured = max(1, int(configured_context_length))
        pending = max(1, int(pending_message_count))
        return min(configured, pending)


class ImageMemoryWritebackService:
    """把已注册聊天消息中的原始图片按组件路径写入 A_Memorix。"""

    def __init__(self, journal_path: Optional[Path] = None) -> None:
        self._journal_path = (
            journal_path if journal_path is not None else ROOT_PATH / "data" / "image_writeback.sqlite3"
        )
        self._journal: Optional[ImageWritebackJournal] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._compensation_task: Optional[asyncio.Task] = None
        self._stopping = False

    async def start(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        if not self._enabled():
            return
        if self._journal is None:
            self._journal = ImageWritebackJournal(self._journal_path)
        self._stopping = False
        self._worker_task = asyncio.create_task(self._worker_loop(), name="A_Memorix.image_writeback")
        self._compensation_task = asyncio.create_task(
            self._compensation_loop(),
            name="A_Memorix.image_description_compensation",
        )

    async def shutdown(self) -> None:
        self._stopping = True
        worker = self._worker_task
        compensation = self._compensation_task
        self._worker_task = None
        self._compensation_task = None
        for task in (worker, compensation):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if self._journal is not None:
            self._journal.close()
            self._journal = None

    async def enqueue(self, message: Any) -> None:
        if not self._enabled() or self._stopping:
            return
        if not any(iter_message_image_components(message.raw_message.components)):
            return
        if self._journal is None:
            await self.start()
        if self._journal is None:
            raise RuntimeError("图片写回服务初始化未完成")
        self._journal.enqueue(str(message.session_id), str(message.message_id))

    @staticmethod
    def _enabled() -> bool:
        """接收与后台处理都服从长期记忆总开关和图片开关。"""
        return global_config.a_memorix.plugin.enabled and global_config.a_memorix.image_memory.enabled

    def list_jobs(self, status: str = "", limit: int = 25, offset: int = 0) -> Dict[str, Any]:
        # 管理服务独立读取任务，不启动消息接收或模型调用。
        journal = ImageWritebackJournal(self._journal_path)
        try:
            return journal.list_jobs(status, limit, offset)
        finally:
            journal.close()

    def retry_failed(self) -> int:
        journal = ImageWritebackJournal(self._journal_path)
        try:
            return journal.retry_failed()
        finally:
            journal.close()

    async def backfill_messages(self, messages: List[Any]) -> Dict[str, int]:
        """同步回填一批历史消息；出现外部引用保证重复执行仍然幂等。"""

        processed = 0
        failed = 0
        for message in messages:
            try:
                if await self._handle_message(message):
                    processed += 1
            except Exception as exc:
                failed += 1
                logger.warning(f"历史图片记忆回填失败: message_id={message.message_id} error={exc}")
        return {"processed_messages": processed, "failed_messages": failed}

    async def _worker_loop(self) -> None:
        while not self._stopping:
            if not self._enabled():
                await asyncio.sleep(max(0.1, global_config.a_memorix.image_memory.job_poll_interval_seconds))
                continue
            if self._journal is None:
                raise RuntimeError("图片写回任务库未初始化")
            job = self._journal.next_job()
            if job is None:
                await asyncio.sleep(max(0.1, global_config.a_memorix.image_memory.job_poll_interval_seconds))
                continue
            try:
                messages = find_messages(session_id=job["session_id"], message_id=job["message_id"], limit=1)
                if not messages:
                    raise ValueError("图片入库任务的来源消息不存在")
                message = messages[0]
                await self._handle_message(message)
                self._journal.complete(job["session_id"], job["message_id"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._journal.fail(job, str(exc), 1 + int(global_config.a_memorix.image_memory.job_max_retries))
                logger.warning(f"图片记忆写入失败，任务已保留: {exc}", exc_info=True)

    async def _compensation_loop(self) -> None:
        """处理持久化的晚到描述任务，进程中断后由租约重新领取。"""

        interval = max(0.1, float(global_config.a_memorix.image_memory.job_poll_interval_seconds))
        while not self._stopping:
            try:
                if self._enabled():
                    await self._process_description_compensations()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"图片描述摘要补偿处理失败: {exc}", exc_info=True)
            await asyncio.sleep(interval)

    async def _process_description_compensations(self) -> None:
        claimed = await memory_service.image_memory(
            action="claim_description_compensations",
            limit=int(global_config.a_memorix.image_memory.job_batch_size),
        )
        if not claimed.get("success"):
            return
        lease_token = str(claimed.get("lease_token") or "")
        for item in claimed.get("items") or []:
            occurrence_id = str(item.get("occurrence_id") or "")
            description_hash = str(item.get("description_hash") or "")
            try:
                result = await memory_service.ingest_summary(
                    external_id=(f"image_description_compensation:{occurrence_id}:{description_hash}"),
                    chat_id=str(item["chat_id"]),
                    text="",
                    participants=[],
                    time_end=float(item["occurred_at"]) + 0.001,
                    metadata={
                        "generate_from_chat": True,
                        "context_length": int(
                            global_config.a_memorix.integration.chat_summary_writeback_context_length
                        ),
                        "writeback_source": "image_description_compensation",
                        "trigger": "image_description_ready",
                        "image_message_id": str(item["message_id"]),
                        "summary_review_count": 2,
                    },
                    respect_filter=True,
                )
                success = result.success
                error = "" if success else result.detail
            except Exception as exc:
                success = False
                error = str(exc)
            await memory_service.image_memory(
                action="complete_description_compensation",
                occurrence_id=occurrence_id,
                description_hash=description_hash,
                lease_token=lease_token,
                success=success,
                error=error,
            )

    @staticmethod
    def _explicit_user_text(components: List[Any]) -> str:
        """只读取用户发送的文本组件，排除 VLM 生成的图片描述。"""

        parts = [component.text.strip() for component in components if isinstance(component, TextComponent)]
        return " ".join(part for part in parts if part)

    async def _handle_message(self, message: Any, *, component_paths: Optional[List[str]] = None) -> bool:
        raw_message = message.raw_message
        components = raw_message.components
        image_components = list(iter_message_image_components(components))
        if not image_components:
            return False
        chat_id = str(message.session_id).strip()
        message_id = str(message.message_id).strip()
        if not chat_id or not message_id:
            raise ValueError("图片消息缺少真实 chat_id 或 message_id")
        timestamp = message.timestamp.timestamp()
        user_statement = self._explicit_user_text(components)
        failures: List[Exception] = []
        for component_path, component in image_components:
            if component_paths is not None and component_path not in component_paths:
                continue
            try:
                if not component.binary_data:
                    await component.load_image_binary()
                result = await memory_service.image_memory(
                    action="ingest",
                    image_bytes=bytes(component.binary_data),
                    source_kind="chat",
                    external_ref=build_chat_external_ref(
                        chat_id=chat_id,
                        message_id=message_id,
                        component_path=component_path,
                    ),
                    scope_type="chat",
                    chat_id=chat_id,
                    message_id=message_id,
                    component_path=component_path,
                    occurred_at=timestamp,
                    user_statement=user_statement,
                )
                from src.chat.image_system.image_manager import image_manager

                if not result.get("success"):
                    raise RuntimeError(str(result.get("error") or "图片记忆写入失败"))

                cached_description = image_manager.get_cached_image_description(component.binary_hash)
                description = (
                    f"[图片：{cached_description}]" if cached_description else str(component.content or "").strip()
                )
                if result.get("success") and description:
                    await memory_service.image_memory(
                        action="describe",
                        content_hash=str(result.get("content_hash") or component.binary_hash),
                        text=description,
                    )
            except Exception as exc:
                exc.add_note(f"图片组件处理失败: message_id={message_id}, component_path={component_path}")
                failures.append(exc)
            finally:
                # 聊天记录只保留图片路径和哈希，需要时可重新载入，避免长期持有原始二进制。
                component.binary_data = b""
        if failures:
            # 处理完所有组件后完整暴露失败，任务保留供重试，已成功组件由外部引用去重。
            raise ExceptionGroup("消息中的图片组件处理失败", failures)
        return True


class MemoryAutomationService:
    def __init__(self) -> None:
        self.fact_writeback = PersonFactWritebackService()
        self.chat_summary_writeback = ChatSummaryWritebackService()
        self.image_writeback = ImageMemoryWritebackService()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self.fact_writeback.start()
        await self.chat_summary_writeback.start()
        await self.image_writeback.start()
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        await self.image_writeback.shutdown()
        await self.chat_summary_writeback.shutdown()
        await self.fact_writeback.shutdown()
        self._started = False

    async def on_incoming_message(self, message: Any) -> None:
        if not self._started:
            await self.start()
        await self._run_message_tasks(
            self._observe_sender_identity(message),
            self.image_writeback.enqueue(message),
            self.fact_writeback.enqueue(message),
        )

    async def on_message_sent(self, message: Any) -> None:
        if not self._started:
            await self.start()
        await self._run_message_tasks(
            self._observe_sender_identity(message),
            self.chat_summary_writeback.enqueue(message),
        )

    @staticmethod
    async def _run_message_tasks(*tasks: Awaitable[None]) -> None:
        """独立完成身份登记和各队列入队，再向调用方完整报告失败。"""
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures: List[Exception] = []
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                failures.append(result)
        if failures:
            raise ExceptionGroup("记忆身份登记或写回入队失败", failures)

    @staticmethod
    async def _observe_sender_identity(message: Any) -> None:
        """在昵称被后续消息覆盖前，按真实发送者 ID 保存本次观察到的称呼。"""
        if not global_config.a_memorix.plugin.enabled:
            return
        user = message.message_info.user_info
        group = message.message_info.group_info
        if not message.platform.strip() or not user.user_id.strip():
            raise ValueError("消息缺少真实的发送者 platform/user_id")
        identity = {
            "platform": message.platform,
            "user_id": user.user_id,
            "aliases": [user.user_nickname, user.user_cardname or ""],
            "group_id": group.group_id if group else "",
        }
        # 短暂的内核初始化或存储故障可以重试；原始 ID 和本次昵称始终来自同一条消息。
        for attempt in range(3):
            try:
                result = await memory_service.profile_admin(action="observe_identity", **identity)
                if not result.get("success"):
                    detail = result.get("error") or result.get("message") or result.get("reason")
                    raise RuntimeError(f"登记发送者身份失败: {detail}")
                return
            except Exception:
                if attempt == 2:
                    raise
                delay = 0.1 * (attempt + 1)
                logger.warning(f"登记发送者身份失败，{delay:.1f} 秒后重试", exc_info=True)
                await asyncio.sleep(delay)


memory_automation_service = MemoryAutomationService()
