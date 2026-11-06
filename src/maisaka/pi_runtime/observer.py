"""Project public Pi events into the existing monitor and usage stores.

These are observations. They never edit SDK history, replay or provider input.
"""

from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List

import time

from src.common.logger import get_logger
from src.maisaka.monitor.events import emit_planner_snapshot
from src.maisaka.monitor import events as monitor_events
from src.services.llm_cache_stats import record_llm_cache_usage
from src.services.pi_harness.accounting import record_pi_usage
from src.services.pi_harness.config import export_harness_config

logger = get_logger("pi_observer")


class PiSubagentObserver:
    """Publish public child activity to its own sidebar, separate from parent rounds."""

    def __init__(self, parent: str, task_id: str) -> None:
        self.parent, self.task_id = parent, task_id
        self.started = time.monotonic()
        self.round_index = 0
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_usage: Dict[str, int] = {}
        self.finished = False

    async def _emit(self, emit: Callable[..., Awaitable[None]], **payload: Any) -> None:
        try:
            await emit(session_id=self.parent, task_id=self.task_id, **payload)
        except Exception:
            # A dashboard outage cannot change the outcome of a child operation.
            logger.exception("Pi 子代理监控记录失败: session=%s task=%s", self.parent, self.task_id)

    async def start(self, description: str, tools: List[str], background: bool, call_id: str) -> None:
        await self._emit(monitor_events.emit_subagent_started, call_id=call_id, description=description,
                         granted_tools=tools, background=background, max_rounds=0)

    async def handle(self, event: Dict[str, Any]) -> None:
        try:
            await self._handle(event)
        except Exception:
            logger.exception("Pi 子代理事件投影失败: session=%s task=%s", self.parent, self.task_id)

    async def _handle(self, event: Dict[str, Any]) -> None:
        message = event.get("message", {})
        if event.get("type") == "message_end" and message.get("role") == "assistant":
            self.round_index += 1
            content = message.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            calls = [{"id": p["id"], "name": p["name"], "arguments": p.get("arguments", {})}
                     for p in content if p.get("type") == "toolCall"]
            for call in calls:
                self.tools[call["id"]] = {"name": call["name"], "round": self.round_index}
            await self._emit(monitor_events.emit_subagent_round, round_index=self.round_index,
                             content="\n".join(p["text"] for p in content if p.get("type") == "text"),
                             tool_calls=calls)
        elif event.get("type") == "tool_execution_start":
            tool = self.tools.get(event["toolCallId"])
            if tool is not None:
                tool["started"] = time.monotonic()
        elif event.get("type") == "message_end" and message.get("role") == "toolResult":
            call_id = message.get("toolCallId", "")
            tool = self.tools.get(call_id)
            if tool is None or tool.get("finished"):
                return
            tool["finished"] = True
            self.tool_usage[tool["name"]] = self.tool_usage.get(tool["name"], 0) + 1
            content = message.get("content", [])
            text = content if isinstance(content, str) else "\n".join(p["text"] for p in content if p.get("type") == "text")
            await self._emit(monitor_events.emit_subagent_tool_result, round_index=tool["round"], call_id=call_id,
                             tool_name=tool["name"], success=not message.get("isError", False),
                             summary=text,
                             duration_ms=max(0, time.monotonic() - tool.get("started", time.monotonic())) * 1000)

    async def finish(self, result: Dict[str, Any]) -> None:
        if self.finished:
            return
        self.finished = True
        await self._emit(monitor_events.emit_subagent_finished,
                         status={"cancelled": "stopped", "interrupted": "stopped"}.get(result["status"], result["status"]),
                         conclusion=result.get("conclusion", ""), error_message=result.get("error") or "",
                         tool_usage=self.tool_usage, duration_ms=max(0, time.monotonic() - self.started) * 1000)


class PiObserver:
    def __init__(self, runtime: Any, *, request_type: str = "maisaka.planner", publish: bool = True) -> None:
        self.runtime = runtime
        self.request_type = request_type
        self.publish = publish
        self.started = 0.0
        self.ended = 0.0
        self.event: Dict[str, Any] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.accounted_entry = None
        self.terminated = False
        self.finished_after_reply = False
        self.delivery_status = ""

    async def handle(self, event: Dict[str, Any]) -> None:
        """Observation failures cannot cancel a provider result or resend a tool."""
        try:
            await self._handle(event)
        except Exception:
            logger.exception("Pi 观察事件记录失败: session=%s type=%s", self.runtime.session_id, event.get("type"))

    async def _handle(self, event: Dict[str, Any]) -> None:
        kind = event["type"]
        self.delivery_status = event.get("deliveryStatus", self.delivery_status)
        message = event.get("message", {})
        if kind == "message_start" and message.get("role") == "assistant":
            self.runtime._cycle_counter += 1
            self.started, self.ended = time.time(), 0.0
            self.event, self.tools = event, {}
            self.terminated = False
            self.finished_after_reply = False
            await self._emit("planner.progress")
        elif kind == "message_end" and message.get("role") == "assistant":
            self.ended = time.time()
            self.event = event
            for part in message.get("content", []):
                if part["type"] == "toolCall":
                    self.tools[part["id"]] = {
                        "tool_call_id": part["id"],
                        "tool_name": part["name"],
                        "tool_args": part.get("arguments", {}),
                        "success": False,
                        "tool_call_source": "response",
                        "tool_call_source_label": "Pi 工具调用",
                    }
            entry_id = event.get("messageEntryId")
            if entry_id and entry_id != self.accounted_entry:
                self.accounted_entry = entry_id
                usage = message.get("usage")
                if isinstance(usage, dict):
                    route = event["route"]
                    await record_pi_usage(
                        {"route": route, "usage": usage},
                        export_harness_config(),
                        datetime.fromtimestamp(self.started),
                        max(0, self.ended - self.started),
                        self.runtime.session_id,
                        task_name="planner",
                        request_type=self.request_type,
                        succeeded=message.get("stopReason") not in {"error", "aborted", "length"},
                    )
                    record_llm_cache_usage(
                        task_name="planner",
                        request_type=self.request_type,
                        model_name=route["modelName"],
                        session_id=self.runtime.session_id,
                        prompt_tokens=int(usage.get("input", 0))
                        + int(usage.get("cacheRead", 0))
                        + int(usage.get("cacheWrite", 0)),
                        prompt_cache_hit_tokens=int(usage.get("cacheRead", 0)),
                        prompt_cache_miss_tokens=int(usage.get("input", 0)) + int(usage.get("cacheWrite", 0)),
                        prompt_text=None,
                    )
            await self._emit("planner.progress")
        elif kind == "tool_execution_start":
            tool = self.tools.get(event["toolCallId"])
            if tool is not None:
                tool["started_at"] = time.time()
            await self._emit("planner.progress", active=event["toolCallId"])
        elif kind == "message_end" and message.get("role") == "toolResult":
            tool = self.tools.get(message["toolCallId"])
            if tool is not None:
                tool["success"] = not message.get("isError", False)
                tool["summary"] = "\n".join(p["text"] for p in message.get("content", []) if p["type"] == "text")
                tool["ended_at"] = time.time()
                tool["duration_ms"] = max(0, tool["ended_at"] - tool.get("started_at", tool["ended_at"])) * 1000
                if tool["tool_name"] == "tool_search":
                    search = message.get("hostSearch", {})
                    tool["matched_tool_names"] = search.get("matched_tool_names", [])
                    tool["newly_discovered_tool_names"] = search.get("newly_discovered_tool_names", [])
            await self._emit("planner.progress")
        elif kind == "turn_end":
            self.terminated = event.get("terminatedByTool", False)
            self.finished_after_reply = event.get("finishedAfterReply", False)
            await self._emit("planner.finalized")

    async def _emit(self, kind: str, *, active: str = "") -> None:
        if not self.publish:
            return
        if not active:
            active = next((
                call_id for call_id, tool in self.tools.items()
                if "started_at" in tool and "ended_at" not in tool
            ), "")
        if kind == "planner.progress" and self.runtime._agent_state == "running":
            tool = self.tools.get(active)
            if tool is None:
                self.runtime._update_stage_status("思考中", "主 Agent 正在判断下一步")
            elif (
                tool["tool_name"] in {"subagent", "read_subagent"}
                and tool["tool_args"].get("wait", tool["tool_name"] == "subagent")
            ):
                self.runtime._update_stage_status("等待子代理", "子代理正在执行，主会话等待结果")
            elif tool["tool_name"] == "reply":
                self.runtime._update_stage_status("发送回复", "正在发送群消息")
            else:
                self.runtime._update_stage_status("工具执行", tool["tool_name"])
        message = self.event.get("message", {})
        usage = message.get("usage", {})
        calls = [p for p in message.get("content", []) if p["type"] == "toolCall"]
        text = "\n".join(p["text"] for p in message.get("content", []) if p["type"] == "text")
        stop = message.get("stopReason", "")
        await emit_planner_snapshot(
            event_type=kind,
            session_id=self.runtime.session_id,
            cycle_id=self.runtime._cycle_counter,
            run_id=self.runtime._monitor_run_id,
            planner_request_messages=None,
            planner_selected_history_count=self.event.get("contextMessageCount"),
            planner_tool_count=self.event.get("toolCount"),
            planner_content=text,
            planner_model_name=self.event.get("route", {}).get("modelName"),
            planner_tool_calls=calls,
            planner_native_tool_calls=[],
            planner_prompt_tokens=int(usage.get("input", 0))
            + int(usage.get("cacheRead", 0))
            + int(usage.get("cacheWrite", 0))
            if usage
            else None,
            planner_completion_tokens=usage.get("output"),
            planner_total_tokens=usage.get("totalTokens"),
            planner_duration_ms=max(0, self.ended - self.started) * 1000 if self.ended else None,
            planner_started_at=self.started or None,
            planner_ended_at=self.ended or None,
            planner_prompt_cache_hit_tokens=usage.get("cacheRead"),
            planner_prompt_cache_miss_tokens=int(usage.get("input", 0)) + int(usage.get("cacheWrite", 0))
            if usage
            else None,
            # The monitor's tools collection contains terminal receipts only;
            # unfinished calls are rendered from planner_tool_calls and active.
            tools=[dict(tool) for tool in self.tools.values() if "ended_at" in tool],
            active_tool_call_id=active,
            agent_state=self.runtime._agent_state,
            planner_interrupted=stop == "aborted",
            end_reason=("provider_" + stop)
            if stop in {"error", "aborted", "length"}
            else ("reply_complete" if self.finished_after_reply else "tool_stop" if self.terminated else "tool_continue" if calls else "planner_no_tool"),
            end_detail="本轮回复发送完成，已结束处理" if self.finished_after_reply else {
                "recovering": "正文已生成但未发送，正在由原 Agent 完成发送选择",
                "undelivered": "正文未发送：一次纠正后仍未调用发送工具",
                "sent": "本轮已有发送工具成功回执",
                "stopped": "本轮已明确选择不发言",
                "waiting": "本轮已明确选择等待",
            }.get(self.delivery_status, "Pi 主会话观察事件"),
        )
