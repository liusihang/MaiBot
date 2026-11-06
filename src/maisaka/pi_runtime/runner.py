"""QQ/business adapter for the official Pi coding-agent harness.

Pi AgentSession/SessionManager own model calls, finalized history, replay and
compaction. Python owns source inbox, tool execution and delivery receipts only.
"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import asyncio
import hashlib
import json
import time
import uuid

from src.common.paths import get_instance_root
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.chat.utils.utils import is_bot_self
from src.config.config import global_config
from src.services.pi_harness.config import export_harness_config
from src.core.tooling import ToolInvocation
from src.llm_models.payload_content.context_item import ContextImagePart, ContextTextPart, UserMessageItem
from src.llm_models.payload_content.tool_option import ToolCall
from src.maisaka.builtin_tool.read_tool_result import get_tool_spec as read_tool_result_spec
from src.maisaka.memory.result_view import format_evidence_page
from src.maisaka.context.messages import (
    ContentContextMessage,
    SessionBackedMessage,
    ToolResultMessage,
    build_context_items_from_history_entry,
)

from .bridge import PiBridge, PiBridgeError
from .business_store import BusinessStore
from .evidence import DurableEvidenceStore, READ_THREAD_ARCHIVE_TOOL_DEFINITION
from .references import build_references
from .observer import PiObserver
from .subagents import PiSubagentService, READ_ONLY_TOOL_NAMES, SUBAGENT_TOOL_DEFINITIONS, format_subagent_result


logger = get_logger("pi_runner")


def pi_tool(definition: Dict[str, Any]) -> Dict[str, Any]:
    function = definition.get("function", definition)
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function.get(
            "parameters", function.get("parameters_schema", {"type": "object", "properties": {}})
        ),
    }


def user_message(text: str, event_id: str) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "timestamp": int(time.time() * 1000),
        "hostEventId": event_id,
    }


def retry_safe_tool(name: str, binding: Dict[str, Any], arguments: Dict[str, Any]) -> bool:
    """Use the frozen execution contract to identify reads and local controls."""
    safety = binding.get("safety") or {}
    owner = binding.get("provider_name"), binding.get("provider_type")
    return (
        owner == ("maibot_pi_internal", "builtin")
        or (owner == ("maisaka_builtin", "builtin") and name in READ_ONLY_TOOL_NAMES | {"stop", "wait"})
        or (safety.get("read_only") is True and safety.get("destructive") is not True)
        or (owner == ("local.gpt-web-search", "plugin") and name in {"gpt_web_search", "gpt_read_search_result"})
        or (owner == ("mcp", "mcp") and name == "browser_navigate"
            and binding.get("routing", {}).get("server_name") == "playwright"
            and binding.get("routing", {}).get("registry_original_name") == name
            and str(arguments.get("url", "")).startswith(("https://", "http://")))
    )


def tool_error(call_id: str, name: str, text: str, outcome: str = "failed") -> Dict[str, Any]:
    """Return an ordinary tool result so the same Agent can choose its next step."""
    return {
        "content": [{"type": "text", "text": text}],
        "details": {"hostReceipt": {"callId": call_id, "success": False}, "execution_outcome": outcome},
        "isError": True, "terminate": False, "finishTurn": False,
    }


def pending_observation(call_id: str, name: str, value: Any) -> Dict[str, Any]:
    """Preserve unknown delivery as an observation, including legacy receipts."""
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        return {**value, "isError": True, "terminate": False, "finishTurn": False,
                "details": {**value.get("details", {}), "execution_outcome": "unknown"}}
    previous = str(value.get("content", "")) if isinstance(value, dict) else ""
    result = tool_error(call_id, name, previous + f"\n工具 {name} 的上次执行结果尚未确认；先核对已有结果，再决定后续操作。", "unknown")
    if isinstance(value, dict):
        metadata = value.get("metadata", value)
        if metadata.get("sent_message_ids"):
            result["details"]["sent_message_ids"] = metadata["sent_message_ids"]
    return result


async def history_message(history: Any, event_id: str) -> Dict[str, Any]:
    """Convert inbound business content, never reconstruct an assistant replay."""
    items = await asyncio.to_thread(build_context_items_from_history_entry, history)
    content = []
    for item in items:
        if not isinstance(item, UserMessageItem):
            continue
        for part in item.parts:
            if isinstance(part, ContextTextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ContextImagePart):
                content.append(
                    {"type": "image", "data": part.image_base64, "mimeType": "image/" + part.normalized_image_format}
                )
    if not content:
        content = [{"type": "text", "text": history.processed_plain_text}]
    payload = {
        "role": "user",
        "content": content,
        "hostEventId": event_id,
        "timestamp": int(history.timestamp.timestamp() * 1000),
    }
    if isinstance(history, SessionBackedMessage):
        source = history.original_message
        if is_bot_self(source.platform, source.message_info.user_info.user_id):
            payload["hostSelfEcho"] = True
    return payload


def source_fingerprint(message: Any) -> str:
    """Hash original source identity/content, not mutable display names or receipt time."""
    identity = {
        "id": str(message.message_id),
        "platform": message.platform,
        "timestamp": message.timestamp.isoformat(),
        "sender": message.message_info.user_info.user_id,
        "group": message.message_info.group_info.group_id if message.message_info.group_info else None,
        "reply_to": message.reply_to,
    }

    def original_component(component: Dict[str, Any]) -> Dict[str, Any]:
        kind = component["type"]
        if kind in {"image", "emoji", "voice"}:
            return {"type": kind, "hash": component["hash"]}
        if kind == "at":
            return {"type": kind, "target_user_id": component["data"]["target_user_id"]}
        if kind == "reply":
            return {"type": kind, "target_message_id": component["data"]}
        if kind == "forward":
            return {
                "type": kind,
                "data": [
                    {
                        "user_id": entry["user_id"],
                        "message_id": entry["message_id"],
                        "content": [original_component(item) for item in entry["content"]],
                    }
                    for entry in component["data"]
                ],
            }
        return component

    # Media descriptions, quote previews and display aliases are derived mutable
    # business annotations; their refresh is not a different original message.
    identity["content"] = [original_component(item) for item in message.raw_message.to_dict()]
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def history_identity(message: Any) -> str | None:
    if isinstance(message, SessionBackedMessage):
        return "input:source:" + message.message_id
    if isinstance(message, ContentContextMessage):
        return "input:" + message.context_item_id
    # Includes immutable EmojiCandidateMessage and plugin-owned user context.
    item = message.to_context_item(enable_visual_message=False)
    return "input:item:" + item.meta.item_id if item is not None else None


class PiGroupRunner:
    def __init__(self, engine: Any, *, data_dir: Path | None = None) -> None:
        self.engine = engine
        self.runtime = engine._runtime
        self.session_id = self.runtime.session_id
        self.data_dir = data_dir if data_dir is not None else get_instance_root() / "data/pi_harness"
        self.ledger = None  # Business receipts only; not a model history ledger.
        self.bridge = PiBridge(
            ["node", str(Path(__file__).resolve().parents[3] / "runtime/pi_agent/sidecar.mjs")], self._callback
        )
        self.subagents = None
        self.evidence = None
        self._initialized = False
        self._model_running = False
        self._tools: Dict[str, Any] = {}
        self._specs: Dict[str, Any] = {}
        self._bindings: Dict[str, Any] = {}
        self._children: Dict[str, Any] = {}
        self._terminated_sessions = set()
        self._orchestration_lock = asyncio.Lock()
        self._maintenance_cancel_lock = asyncio.Lock()
        self._maintenance_task: asyncio.Task[None] | None = None
        self._storage_lock = asyncio.Lock()
        self._initialization_lock = asyncio.Lock()
        self._ingress_lock = asyncio.Lock()
        self._tool_media_ids = set()
        self._failed_compaction_head: str | None = None
        self.observer = PiObserver(self.runtime)

    async def _db(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(getattr(self.ledger, method), *args, **kwargs)

    async def _ensure_storage(self) -> None:
        async with self._storage_lock:
            if self.ledger is None:
                await asyncio.to_thread(self.data_dir.mkdir, parents=True, exist_ok=True)
                self.ledger = await asyncio.to_thread(BusinessStore, self.data_dir / "business.sqlite3")
                self.subagents = PiSubagentService(
                    self.data_dir / "children.sqlite3", self._run_child, self._deliver_child_completion
                )
                self.evidence = DurableEvidenceStore(self.data_dir / "evidence.sqlite3")

    async def record_pending(self, message: Any) -> None:
        await self._interrupt_context_maintenance()
        await self._ensure_storage()
        event_id = "input:source:" + str(message.message_id)
        async with self._ingress_lock:
            fingerprint = await asyncio.to_thread(source_fingerprint, message)
            await self._db("validate_source", self.session_id, event_id, fingerprint)
            if await self._db("has_input", self.session_id, event_id):
                return
            history = await self.engine._build_history_message(message)
            if history is None:
                raise ValueError("Inbound message could not be frozen for Pi")
            payload = await history_message(history, event_id)
            payload["sourceMessageId"] = str(message.message_id)
            await self._db("enqueue_input", self.session_id, event_id, payload)

    async def pending_source_ids(self) -> List[str]:
        await self._ensure_storage()
        return [
            event.payload["sourceMessageId"]
            for event in await self._db("pending_inputs", self.session_id)
            if "sourceMessageId" in event.payload
        ]

    async def ensure_initialized(self) -> None:
        async with self._initialization_lock:
            if self._initialized and self.bridge.is_running:
                return
            if self._initialized:
                # A new entry may reopen the official durable session after a
                # dead sidecar. Never replay the failed request or tool effect.
                self._initialized = False
                await self.bridge.close()
            await self._initialize()

    async def _initialize(self) -> None:
        await self._ensure_storage()
        service = self.runtime._chat_loop_service
        service.passive_replyer_override = False
        self.engine._turn_passive_replyer = False
        specs = await self.runtime._tool_registry.list_tools(self.engine._build_tool_availability_context())
        if len({spec.name for spec in specs}) != len(specs):
            raise RuntimeError("Duplicate tool names cannot bind a durable Pi session")
        self._specs = {spec.name: spec for spec in specs}
        for spec in specs:
            if spec.name not in {"subagent", "subagent_status", "subagent_stop"}:
                self._tools[spec.name] = pi_tool(spec.to_llm_definition())
                self._bindings[spec.name] = spec.execution_contract()
        for definition in [
            *SUBAGENT_TOOL_DEFINITIONS,
            READ_THREAD_ARCHIVE_TOOL_DEFINITION,
            read_tool_result_spec().to_llm_definition(),
        ]:
            tool = pi_tool(definition)
            self._tools[tool["name"]] = tool
            self._bindings[tool["name"]] = {"provider_name": "maibot_pi_internal", "provider_type": "builtin"}
        self.runtime.set_current_action_tool_names(list(self._tools))
        await self.bridge.start()
        state = await self.bridge.request(
            "init",
            sessionId=self.session_id,
            sessionDir=str(self.data_dir / "sessions"),
            system=service._build_chat_system_prompt(),
            tools=list(self._tools.values()),
            toolBindings=self._bindings,
            config=export_harness_config(),
            taskName="planner",
            requireDelivery=True,
        )
        await self._recover_tool_results(state)
        self._initialized = True
        await self.subagents.recover(self.session_id)
        pending = await self._db("pending_inputs", self.session_id)
        turn = await self._db("pending_turn", self.session_id)
        if turn and state.get("completedRequestId") == turn["turn_id"]:
            await self._db("complete_turn", self.session_id, turn["turn_id"])
            turn = None
        if any(event.payload.get("piChildCompletion") for event in pending):
            self._wake_for_child_completion()

    async def restore_pending_turn(self) -> None:
        """Schedule once at host startup, not during an already active run's reconnect."""
        await self.ensure_initialized()
        turn = await self._db("pending_turn", self.session_id)
        if turn is None:
            return
        state = await self.bridge.request("state", sessionId=self.session_id)
        await self._recover_tool_results(state)
        self._wake_for_recovery(turn["turn_id"], turn["source_ids"])

    async def _recover_tool_results(self, state: Dict[str, Any]) -> None:
        """Complete interrupted native calls with observations, never redispatch them."""
        results = []
        missing = {call["id"] for call in state.get("unresolvedToolCalls", [])}
        calls = {call["id"]: call for call in [*state.get("pendingToolObservations", []), *state.get("unresolvedToolCalls", [])]}
        for call in calls.values():
            receipt = await self._db("lookup_tool_execution", self.session_id, call["id"])
            if receipt is not None and receipt.status in {"succeeded", "failed"}:
                value = receipt.result
            else:
                intent = json.loads(receipt.intent_json) if receipt is not None else {}
                safe = retry_safe_tool(call["name"], self._bindings.get(call["name"], {}), intent.get("arguments", {}))
                if receipt is None or receipt.status == "prepared" or safe:
                    value = tool_error(call["id"], call["name"],
                                       f"工具 {call['name']} 的上次调用已中断，可根据当前情况重新选择操作。",
                                       "not_executed" if receipt is None or receipt.status == "prepared" else "failed")
                    if receipt is not None:
                        if receipt.status == "prepared":
                            await self._db("claim_tool_execution", self.session_id, call["id"])
                        await self._db("record_tool_outcome", self.session_id, call["id"], value, status="failed")
                else:
                    if call["id"] not in missing and receipt.status == "unknown":
                        continue
                    value = pending_observation(call["id"], call["name"], receipt.result)
                    await self._db("mark_tool_execution_unknown", self.session_id, call["id"], observation=value)
                    if call["id"] not in missing:
                        # This observation is already in the native context.
                        # Keep checking for a future confirmed receipt without reinjecting it.
                        continue
            results.append({"role": "toolResult", "toolCallId": call["id"], "toolName": call["name"],
                            "content": value["content"], "details": value.get("details", {}),
                            "isError": value["isError"], "timestamp": int(time.time() * 1000)})
        if results:
            await self.bridge.request("resolve_tools", sessionId=self.session_id, results=results)

    def _wake_for_recovery(self, turn_id: str, source_ids: List[str]) -> None:
        task_id = "pi-recover:" + turn_id
        description = (
            f"继续重启前未完成的群聊任务，原始消息 ID：{', '.join(source_ids) or '见线程中未完成的任务'}。"
            "这是内部恢复事件，不是新的群友发言。结合当前时间和最新消息判断如何接续；"
            "回复引用真实原始消息，已成功发送的内容不再发送，工具结果未知时不重试。"
        )
        self.runtime._arm_forced_turn_state(message_id=task_id, reason="恢复未完成群聊任务")
        self.runtime._queue_proactive_turn(self.runtime._build_proactive_trigger_message(task_id, description))
        logger.info(f"[{self.session_id}] 已调度恢复未完成轮次: {turn_id}")

    def _wake_for_child_completion(self) -> None:
        """Reuse the existing forced internal-turn scheduler, bypassing Luna cooldown."""
        task_id = "pi-subagent-completion:" + self.session_id
        self.runtime._arm_forced_turn_state(message_id=task_id, reason="子 Agent 任务完成")
        self.runtime._queue_proactive_turn(self.runtime._build_proactive_trigger_message(task_id, "子 Agent 任务完成"))

    async def _deliver_child_completion(self, result: Dict[str, Any]) -> None:
        await self._interrupt_context_maintenance()
        event_id = "child-complete:" + result["handle"]
        if not await self._db("has_input", self.session_id, event_id):
            payload = user_message(
                "<subagent_completion>\n" + format_subagent_result(result) + "\n</subagent_completion>",
                event_id,
            )
            payload["piChildCompletion"] = True
            await self._db("enqueue_input", self.session_id, event_id, payload)
        if not self._model_running:
            self._wake_for_child_completion()

    async def ingest(self, *, source_message_ids: set[str] | None = None) -> None:
        await self._interrupt_context_maintenance()
        await self.ensure_initialized()
        learning_sources = []
        # Existing host mirror remains for QQ quoting/UI, not assistant replay.
        for message in list(self.runtime._chat_history):
            if isinstance(message, ToolResultMessage) and message.tool_name == "wait":
                event_id = "wait-complete:" + message.context_item_id
                payload = user_message("Wait completion event: " + message.content, event_id)
                payload["timestamp"] = int(message.timestamp.timestamp() * 1000)
            elif message.role == "user":
                event_id = await asyncio.to_thread(history_identity, message)
                if event_id is None or event_id in self._tool_media_ids:
                    continue
                if await self._db("is_tool_media_input", self.session_id, event_id):
                    continue
                if isinstance(message, SessionBackedMessage):
                    if source_message_ids is not None and message.message_id not in source_message_ids:
                        continue
                    learning_sources.append(message)
                    fingerprint = await asyncio.to_thread(source_fingerprint, message.original_message)
                    await self._db("validate_source", self.session_id, event_id, fingerprint)
                    if await self._db("has_input", self.session_id, event_id):
                        continue
                payload = await history_message(message, event_id)
                if isinstance(message, SessionBackedMessage):
                    payload["sourceMessageId"] = message.message_id
            else:
                continue
            # Idempotence always compares content; never silently swallow a
            # changed internal source just because its ID already exists.
            await self._db("enqueue_input", self.session_id, event_id, payload)
        pending = await self._db("pending_inputs", self.session_id)
        if source_message_ids is not None:
            pending = [
                event
                for event in pending
                if not event.payload.get("sourceMessageId") or event.payload["sourceMessageId"] in source_message_ids
            ]
        if pending:
            result = await self.bridge.request(
                "append", sessionId=self.session_id, additions=[event.payload for event in pending]
            )
            for event_id in result.get("acceptedSourceIds", []):
                await self._db("acknowledge_input", self.session_id, event_id)
        # Learning consumes original business sources, never Pi assistant state or
        # a Python-generated summary. It is independent of whether Luna engages.
        if learning_sources:
            await self.runtime._trigger_pi_source_learning(learning_sources)

    async def compact(self, *, summary: str | None = None) -> Dict[str, Any]:
        """Explicit SDK compaction; no Python transcript or summary generation."""
        await self._interrupt_context_maintenance()
        async with self._orchestration_lock:
            await self.ensure_initialized()
            state = await self.bridge.request("state", sessionId=self.session_id)
            arguments = {"sessionId": self.session_id, "expectedRevision": state["revision"]}
            if summary is not None:
                arguments["summary"] = summary
            try:
                return await self.bridge.request("compact", **arguments)
            except asyncio.CancelledError:
                await asyncio.shield(self.bridge.request("abort", sessionId=self.session_id))
                raise

    async def _interrupt_context_maintenance(self) -> None:
        """New conversation work takes priority over the idle summary request."""
        async with self._maintenance_cancel_lock:
            task = self._maintenance_task
            if task is None:
                return
            if not task.done() and not task.cancelling():
                task.cancel()
            # Cancellation waits for the SDK summary to settle before another
            # main run acquires the session. Concurrent intake must not cancel
            # that cleanup a second time or abort a later main model request.
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))
            if self._maintenance_task is task:
                self._maintenance_task = None

    def _schedule_context_maintenance(self) -> None:
        if self._maintenance_task is None or self._maintenance_task.done():
            self._maintenance_task = asyncio.create_task(
                self._run_context_maintenance(), name="pi-context-maintenance:" + self.session_id
            )

    async def _run_context_maintenance(self) -> None:
        async with self._orchestration_lock:
            if self.runtime._has_pending_messages():
                return
            try:
                await self._maintain_context()
            except PiBridgeError as exc:
                if exc.code in {"SUMMARY_FAILED", "INVALID_SUMMARY", "REVISION_CONFLICT"} or (
                    "Pi summary request did not complete" in str(exc)
                ):
                    logger.warning(f"[{self.session_id}] 本轮已完成；上下文摘要未完成，原始历史保留: {exc}")
                else:
                    logger.exception(f"[{self.session_id}] 后台上下文压缩失败，原始历史保留")
            except Exception:
                logger.exception(f"[{self.session_id}] 后台上下文压缩失败，原始历史保留")

    async def _maintain_context(self) -> None:
        state = await self.bridge.request("state", sessionId=self.session_id)
        tokens = state["contextTokens"]
        if tokens < global_config.chat.context_prompt_token_high_watermark:
            return
        # These are pressure signals, not a tokenizer or hard capacity proof.
        # No retry of the same unchanged failed batch and no fallback trimming.
        head = state["archive"]["head"]
        if head == self._failed_compaction_head:
            return
        try:
            await self.bridge.request(
                "compact",
                sessionId=self.session_id,
                expectedRevision=state["revision"],
                preserveUnprocessed=True,
                summaryTaskName="memory",
                summaryTimeoutMs=300000,
            )
        except asyncio.CancelledError:
            # This task owns the orchestration lock, so abort can only stop its
            # summary. An interrupted summary is eligible again after chatting.
            await asyncio.shield(self.bridge.request("abort", sessionId=self.session_id))
            logger.info(f"[{self.session_id}] 后台压缩已中止，优先处理新消息；原始上下文保留")
            raise
        except BaseException:
            self._failed_compaction_head = head
            raise
        self._failed_compaction_head = None

    async def run(self, source_messages: List[Any]) -> None:
        await self._interrupt_context_maintenance()
        async with self._orchestration_lock:
            await self.ensure_initialized()
            await self._recover_tool_results(await self.bridge.request("state", sessionId=self.session_id))
            turn = await self._db(
                "begin_turn", self.session_id, [str(message.message_id) for message in source_messages]
            )
            focus_group = getattr(self.runtime, "_focus_group", None)
            await self.ingest(
                source_message_ids={message.message_id for message in source_messages}
                if focus_group is not None
                else None
            )
            from src.services.daily_topics.service import daily_topic_service

            daily_references = await daily_topic_service.reference(self.session_id)
            references = [
                {"namespace": ref.namespace, "sourceId": ref.source_id, "version": ref.version, "content": ref.content}
                for ref in [*await build_references(self.runtime, source_messages), *daily_references]
            ]
            self.engine._active_logical_turn_id = uuid.uuid4().hex
            self.runtime._start_planner_continuation()
            self._model_running = True
            self.runtime._update_stage_status("思考中", "等待主 Agent 响应")
            try:
                state = await self.bridge.request(
                    "prompt",
                    sessionId=self.session_id,
                    requestId=turn["turn_id"],
                    references=references,
                    additions=[
                        user_message(
                            "Current time: " + datetime.now().astimezone().isoformat(), "time:" + uuid.uuid4().hex
                        )
                    ],
                )
                if state.get("paused") or state.get("error"):
                    raise RuntimeError(
                        "Pi session paused: " + str(state.get("error") or state.get("unresolvedToolCalls"))
                    )
                if state.get("completedRequestId") != turn["turn_id"]:
                    raise RuntimeError("Pi request returned without durable completion")
                await self._db("complete_turn", self.session_id, turn["turn_id"])
            except asyncio.CancelledError:
                # Cancelling the Python wait must stop the SDK run as well. Wait
                # for its terminal persistence/unknown-effect marker before exit.
                abort = asyncio.create_task(self.bridge.request("abort", sessionId=self.session_id))
                await asyncio.shield(abort)
                raise
            finally:
                self._model_running = False
                self.runtime._end_planner_continuation()
                self.engine._active_logical_turn_id = None
                if self.runtime._agent_state == self.runtime._STATE_RUNNING:
                    self.runtime._enter_stop_state()
                self.runtime._update_stage_status("空闲", "Pi 主线程等待下一批群消息")
                if self.runtime._running:
                    self.runtime._schedule_message_turn()
                pending = await self._db("pending_inputs", self.session_id)
                if any(event.payload.get("piChildCompletion") for event in pending):
                    self._wake_for_child_completion()

            # Return control to the message loop after durable completion. Idle
            # maintenance is interruptible and never gates the next chat turn.
            self._schedule_context_maintenance()

    async def _callback(self, method: str, params: Dict[str, Any]) -> Any:
        session_id = params["sessionId"]
        if session_id != self.session_id and session_id not in self._children:
            raise PermissionError("Pi callback belongs to an unknown session")
        if method == "authorize":
            # A lost IPC ACK does not mean the external tool failed. Supply a
            # durable receipt before the SDK turns a bridge error into a retry.
            if params.get("toolCallId"):
                receipt = await self._db("lookup_tool_execution", session_id, params["toolCallId"])
                if (receipt is not None and receipt.status in {"succeeded", "failed"}
                        and json.loads(receipt.intent_json).get("name") == params.get("name")):
                    return {"allowed": True, "toolResult": receipt.result}
            return {"allowed": True}
        if method == "tool":
            return await self._tool(params)
        if method != "event":
            raise ValueError("Python does not implement model inference callbacks: " + method)
        event = params["event"]
        child = self._children.get(session_id)
        if child is None:
            await self.observer.handle(event)
        if child is not None:
            await child["observer"].handle(event)
            await child["on_event"](event)
            if event["type"] == "turn_end" and child["pending"]:
                additions, child["pending"] = child["pending"], []
                await self.bridge.request("append", sessionId=session_id, additions=additions)
        elif event["type"] == "turn_end" and getattr(self.runtime, "_focus_group", None) is None:
            # Focus groups admit only their ready snapshot, not arrivals still
            # in debounce. Ordinary group arrivals enter at this safe boundary.
            # This is business intake, not a durability ACK for model history.
            if self.runtime._has_pending_messages():
                messages = await self.engine._ingest_pending_messages()
                stopping = any(
                    part.get("type") == "toolCall" and part.get("name") == "stop"
                    for part in event.get("message", {}).get("content", [])
                )
                if stopping and any(
                    not is_bot_self(message.platform, message.message_info.user_info.user_id) for message in messages
                ):
                    self.runtime._enter_running_state()
            await self.ingest()
        return {}

    async def _tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        child = self._children.get(params["sessionId"])
        name, call_id, arguments = params["name"], params["toolCallId"], params["arguments"]
        if name not in self._tools or name not in self._bindings:
            return tool_error(call_id, name, "Tool is not bound to this durable Pi thread; nothing was executed.", "not_executed")
        binding = self._bindings[name]
        if binding["provider_name"] != "maibot_pi_internal":
            current_specs = await self.runtime._tool_registry.list_tools(self.engine._build_tool_availability_context())
            matches = [spec for spec in current_specs if spec.name == name]
            if len(matches) != 1:
                return tool_error(call_id, name, "Tool is missing or ambiguous; nothing was executed.", "not_executed")
            current = matches[0]
            if pi_tool(current.to_llm_definition()) != self._tools[name] or current.execution_contract() != binding:
                # Rejected before claiming an effect; this is not an unknown send.
                return tool_error(call_id, name, "Tool schema or provider changed; nothing was executed.", "not_executed")
        if child:
            try:
                PiSubagentService.assert_tool_allowed(name, child["tools"])
            except PermissionError as exc:
                return tool_error(call_id, name, str(exc), "not_executed")
        session_id = params["sessionId"]
        intent = {"name": name, "arguments": arguments}
        receipt = await self._db("prepare_tool_execution", session_id, call_id, intent)
        if receipt.status in {"succeeded", "failed"}:
            if receipt.result.get("terminate"):
                self._terminated_sessions.add(session_id)
            return receipt.result
        if receipt.status in {"executing", "unknown"}:
            return pending_observation(call_id, name, receipt.result)
        if not await self._db("claim_tool_execution", session_id, call_id):
            return tool_error(call_id, name, "这次工具调用已经开始执行，请先核对已有结果。", "unknown")
        if not retry_safe_tool(name, binding, arguments):
            pending = await self._db("find_pending_effect", session_id, call_id, intent)
            if pending is not None:
                result = tool_error(call_id, name, "相同操作此前的执行结果尚未确认，本次未重复执行；可以继续其他操作。", "not_executed")
                result["details"]["pendingCallId"] = pending.call_id
                await self._db("record_tool_outcome", session_id, call_id, result, status="failed")
                return result
        media_content = []
        media_event_ids = []
        search_observation = None
        post_execution = None
        finish_turn = False
        uncertain = False
        provider_result = None
        try:
            if name == "subagent":
                value = await self.subagents.delegate(
                    self.session_id, arguments["task"], list(self._tools.values()),
                    wait=arguments.get("wait", True), call_id=call_id,
                )
                text = format_subagent_result(value)
                success, terminate = value["status"] in {"completed", "running"}, False
            elif name in {"read_subagent", "read_tool_result", "read_thread_archive"}:
                try:
                    if name == "read_subagent":
                        value = await self.subagents.read(
                            self.session_id,
                            arguments["handle"],
                            arguments.get("offset", 0),
                            arguments.get("limit", 12000),
                            wait=arguments.get("wait", False),
                        )
                    elif name == "read_thread_archive":
                        from src.services.daily_topics.service import SOURCE_PREFIX, daily_topic_service

                        if any(str(item).startswith(SOURCE_PREFIX) for item in arguments["event_ids"]):
                            value = await daily_topic_service.read_sources(
                                self.session_id,
                                arguments["event_ids"],
                                arguments.get("offset", 0),
                                arguments.get("limit", 2800),
                            )
                        else:
                            value = await self.bridge.request(
                                "read_archive",
                                sessionId=self.session_id,
                                eventIds=arguments["event_ids"],
                                offset=arguments.get("offset", 0),
                                limit=arguments.get("limit", 2800),
                            )
                    else:
                        value = await self.evidence.read(
                            self.session_id,
                            arguments["result_id"],
                            arguments.get("offset", 0),
                            arguments.get("max_chars", 2000),
                            arguments.get("keyword", ""),
                        )
                    if name == "read_subagent":
                        text = format_subagent_result(value)
                    elif name == "read_tool_result":
                        text = format_evidence_page(value, arguments["result_id"])
                    else:
                        text = json.dumps(value, ensure_ascii=False)
                    success = True
                except PiBridgeError as exc:
                    if exc.code not in {"ARCHIVE_NOT_FOUND", "INVALID_ARCHIVE_QUERY", "INVALID_PAGINATION"}:
                        raise
                    text, success = "Evidence read rejected: " + exc.code, False
                except (PermissionError, ValueError, KeyError) as exc:
                    # These readers have no external effects. A known invalid
                    # handle/argument is a normal failed result, not unknown send.
                    text, success = f"Evidence read rejected: {type(exc).__name__}: {exc}", False
                terminate = False
            elif (
                child is not None
                and name in {"stop", "wait"}
                and binding["provider_name"] == "maisaka_builtin"
                and binding["provider_type"] == "builtin"
            ):
                # These controls belong to the child; invoking the host handlers
                # would stop or suspend the parent chat's business scheduler.
                if name == "wait":
                    await asyncio.sleep(max(0, arguments["seconds"]))
                text, success, terminate = "Child control completed", True, name == "stop"
            else:
                invocation = ToolInvocation(
                    tool_name=name,
                    arguments=self.runtime.expand_message_id_aliases(dict(arguments)),
                    call_id=call_id,
                    session_id=self.session_id,
                    stream_id=self.session_id,
                    reasoning="",
                )
                focus_group = getattr(self.runtime, "_focus_group", None)
                if (
                    child is None
                    and focus_group is not None
                    and name in {"wait", "stop"}
                    and binding["provider_name"] == "maisaka_builtin"
                    and binding["provider_type"] == "builtin"
                ):
                    # Preserve Focus scheduling controls without a shared model
                    # context. Revalidate after receipt-claim awaits, then mutate
                    # only the authorized business scheduler and checkpoint it.
                    current = await self.runtime._tool_registry.list_tools(
                        self.engine._build_tool_availability_context()
                    )
                    matches = [spec for spec in current if spec.name == name]
                    if len(matches) != 1 or matches[0].execution_contract() != binding:
                        raise PermissionError("Frozen Focus control binding changed before dispatch")
                    result = focus_group.execute_control(name, invocation.arguments, self.session_id)
                    await focus_group.checkpoint()
                else:
                    result = await self.runtime._tool_registry.invoke(
                        invocation,
                        self.engine._build_tool_execution_context(""),
                        expected_contract=binding,
                    )
                provider_result = result
                # Providers return ordinary failures for model self-correction.
                # Retain explicit uncertainty/partial sends only as operation-level receipts.
                uncertain = not result.success and not retry_safe_tool(name, binding, arguments) and (
                    result.metadata.get("execution_outcome") == "unknown"
                    or bool(result.metadata.get("sent_message_ids"))
                )
                evidence_handle = await self.evidence.capture(
                    self.session_id, result, child_namespace=session_id if child else ""
                )
                if child is not None and result.success:
                    source = {"kind": "host_tool_receipt", "tool": name, "call_id": call_id, "result_id": evidence_handle}
                    # The search plugin owns its durable originals and reports
                    # their reader in its JSON body rather than host metadata.
                    if name in {"gpt_web_search", "gpt_read_search_result"} and evidence_handle is None:
                        try:
                            search = json.loads(result.get_history_content())
                        except (TypeError, ValueError):
                            search = None
                        if isinstance(search, dict) and isinstance(search.get("result_id"), str):
                            source.update(result_id=search["result_id"], reader_tool="gpt_read_search_result")
                    child["sources"].append(source)
                if child is None:
                    post_execution = (invocation, result, self._specs.get(name))
                text, success = result.get_history_content(), result.success
                if (
                    child is None and name == "reply" and success
                    and binding["provider_name"] == "maisaka_builtin" and binding["provider_type"] == "builtin"
                ):
                    finish_turn = result.metadata.get("finish_turn") is True
                if name == "tool_search":
                    search_observation = {
                        key: result.metadata.get(key, [])
                        for key in ("matched_tool_names", "newly_discovered_tool_names")
                    }
                terminate = result.stop_after_execution or bool(result.metadata.get("pause_execution"))
                media = self.engine._build_tool_result_media_messages(ToolCall(call_id, name, arguments), result)
                for index, extra in enumerate([*result.post_history_messages, *media]):
                    payload = await history_message(extra, f"tool-media:{call_id}:{index}")
                    media_content.extend(payload["content"])
                    if child is None:
                        # Keep host UI/emoji business capabilities, but the Pi
                        # toolResult owns this exact media snapshot from now on.
                        event_id = await asyncio.to_thread(history_identity, extra)
                        if event_id is not None:
                            self._tool_media_ids.add(event_id)
                            media_event_ids.append(event_id)
                        self.runtime._chat_history.append(extra)
            callback_result = {
                "content": [{"type": "text", "text": text or "(empty result)"}, *media_content],
                "details": {
                    "hostReceipt": {"callId": call_id, "success": success},
                    "hostMediaEventIds": media_event_ids,
                    **({"hostSearch": search_observation} if search_observation is not None else {}),
                },
                "isError": not success,
                "terminate": terminate,
                "finishTurn": finish_turn,
            }
            if uncertain:
                callback_result["content"].append({"type": "text", "text": "本次执行结果尚未确认；先核对已有结果，再决定后续操作。"})
                callback_result["details"]["execution_outcome"] = "unknown"
                if result.metadata.get("sent_message_ids"):
                    callback_result["details"]["sent_message_ids"] = result.metadata["sent_message_ids"]
                await self._db("mark_tool_execution_unknown", session_id, call_id, observation=callback_result)
            else:
                if terminate:
                    self._terminated_sessions.add(session_id)
                await self._db(
                    "record_tool_outcome", session_id, call_id, callback_result, status="succeeded" if success else "failed"
                )
            if post_execution is not None:
                try:
                    await self.engine._record_tool_execution_effects(*post_execution)
                except Exception:
                    # Monitor/history/feedback writes observe a completed tool;
                    # they cannot turn its durable receipt into an unknown send.
                    logger.exception(f"[{session_id}] 工具回执已保存，后台后处理失败: {name} {call_id}")
            return callback_result
        except asyncio.CancelledError:
            receipt = await asyncio.shield(self._db("lookup_tool_execution", session_id, call_id))
            if receipt is not None and receipt.status in {"executing", "unknown"}:
                await asyncio.shield(self._db("mark_tool_execution_unknown", session_id, call_id))
            raise
        except Exception as exc:
            receipt = await self._db("lookup_tool_execution", session_id, call_id)
            if receipt is not None and receipt.status in {"succeeded", "failed"}:
                return receipt.result
            if provider_result is not None:
                # Evidence/media bookkeeping cannot change a known provider
                # outcome into an uncertain send or cause an already sent reply to retry.
                success = provider_result.success
                value = {
                    "content": [{"type": "text", "text": provider_result.get_history_content()}, *media_content],
                    "details": {"hostReceipt": {"callId": call_id, "success": success},
                                "hostMediaEventIds": media_event_ids},
                    "isError": not success,
                    "terminate": provider_result.stop_after_execution or bool(provider_result.metadata.get("pause_execution")),
                    "finishTurn": success and name == "reply"
                    and binding.get("provider_name") == "maisaka_builtin" and binding.get("provider_type") == "builtin"
                    and provider_result.metadata.get("finish_turn") is True,
                }
                if provider_result.metadata.get("sent_message_ids"):
                    value["details"]["sent_message_ids"] = provider_result.metadata["sent_message_ids"]
                if not success and not retry_safe_tool(name, binding, arguments) and (
                    provider_result.metadata.get("execution_outcome") == "unknown"
                    or bool(provider_result.metadata.get("sent_message_ids"))
                ):
                    value["details"]["execution_outcome"] = "unknown"
                    await self._db("mark_tool_execution_unknown", session_id, call_id, observation=value)
                else:
                    await self._db("record_tool_outcome", session_id, call_id, value,
                                   status="succeeded" if success else "failed")
                logger.exception(f"[{session_id}] 工具结果已保留，附加资料处理失败: {name} {call_id}")
                return value
            value = tool_error(call_id, name, f"工具 {name} 调用失败：{type(exc).__name__}: {exc}",
                               "failed" if retry_safe_tool(name, binding, arguments) else "unknown")
            if value["details"]["execution_outcome"] == "unknown":
                value["content"].append({"type": "text", "text": "本次执行结果尚未确认；先核对已有结果，再决定后续操作。"})
                await self._db("mark_tool_execution_unknown", session_id, call_id, observation=value)
            else:
                await self._db("record_tool_outcome", session_id, call_id, value, status="failed")
            logger.exception(f"[{session_id}] 工具失败已返回 Agent 继续处理: {name} {call_id}")
            return value

    async def _run_child(
        self, child_id: str, parent: str, task: str, tools: List[Any], on_event: Any
    ) -> Dict[str, Any]:
        self._children[child_id] = {
            "tools": [pi_tool(tool)["name"] for tool in tools],
            "on_event": on_event,
            "pending": [],
            "sources": [],
            "observer": PiObserver(
                SimpleNamespace(session_id=self.session_id, _cycle_counter=0),
                request_type="maisaka.sub_agent",
                publish=False,
            ),
        }
        try:
            await self.bridge.request(
                "init",
                sessionId=child_id,
                system=load_prompt(
                    "maisaka_subagent",
                    current_time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                ),
                tools=[pi_tool(tool) for tool in tools],
                sessionDir=str(self.data_dir / "sessions"),
                config=export_harness_config(),
                taskName="planner",
                toolBindings={pi_tool(tool)["name"]: self._bindings[pi_tool(tool)["name"]] for tool in tools},
            )
            state = await self.bridge.request(
                "prompt", sessionId=child_id, additions=[user_message(task, "child-task:" + child_id)]
            )
            if state.get("paused") or state.get("error"):
                unresolved = ", ".join(call["name"] for call in state.get("unresolvedToolCalls", []))
                reason = state.get("error") or "Unresolved child tool outcome: " + unresolved
                raise PiBridgeError(str(reason), code="CHILD_EXECUTION_FAILED")
            return {**state, "sources": list(self._children[child_id]["sources"])}
        finally:
            try:
                await self.bridge.request("close", sessionId=child_id)
            finally:
                self._children.pop(child_id, None)

    async def close(self) -> None:
        async def settle_sessions() -> None:
            await self._interrupt_context_maintenance()
            if self.subagents is not None:
                await self.subagents.close()
            if self._initialized and self.bridge.is_running:
                await self.bridge.request("close", sessionId=self.session_id)
            for child_id in list(self._children):
                if child_id in self._children and self.bridge.is_running:
                    await self.bridge.request("close", sessionId=child_id)

        shutdown = asyncio.create_task(settle_sessions())
        try:
            await asyncio.shield(shutdown)
        except asyncio.CancelledError:
            # Terminal events/outbox outcomes are durability barriers, not an
            # optional clean-up canceled together with the outer runtime task.
            await shutdown
            raise
        finally:
            await self.bridge.close()
            self._initialized = False
