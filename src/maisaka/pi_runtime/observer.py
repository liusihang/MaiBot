"""Project public Pi events into the existing monitor and usage stores.

These are observations. They never edit SDK history, replay or provider input.
"""

from datetime import datetime
from typing import Any, Dict

import time

from src.common.logger import get_logger
from src.maisaka.monitor.events import emit_planner_snapshot
from src.services.llm_cache_stats import record_llm_cache_usage
from src.services.pi_harness.accounting import record_pi_usage
from src.services.pi_harness.config import export_harness_config

logger = get_logger("pi_observer")


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

    async def handle(self, event: Dict[str, Any]) -> None:
        """Observation failures cannot cancel a provider result or resend a tool."""
        try:
            await self._handle(event)
        except Exception:
            logger.exception("Pi 观察事件记录失败: session=%s type=%s", self.runtime.session_id, event.get("type"))

    async def _handle(self, event: Dict[str, Any]) -> None:
        kind = event["type"]
        message = event.get("message", {})
        if kind == "message_start" and message.get("role") == "assistant":
            self.runtime._cycle_counter += 1
            self.started, self.ended = time.time(), 0.0
            self.event, self.tools = event, {}
            self.terminated = False
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
            await self._emit("planner.finalized")

    async def _emit(self, kind: str, *, active: str = "") -> None:
        if not self.publish:
            return
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
            tools=list(self.tools.values()),
            active_tool_call_id=active,
            agent_state=self.runtime._agent_state,
            planner_interrupted=stop == "aborted",
            end_reason=("provider_" + stop)
            if stop in {"error", "aborted", "length"}
            else ("tool_stop" if self.terminated else "tool_continue" if calls else "planner_no_tool"),
            end_detail="Pi 主会话观察事件",
        )
