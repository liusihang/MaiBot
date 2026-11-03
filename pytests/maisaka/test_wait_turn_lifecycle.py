from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import asyncio
import time

import pytest

from src.maisaka.reasoning_engine import MaisakaReasoningEngine
from src.maisaka.runtime import MaisakaHeartFlowChatting


def _build_runtime_stub() -> MaisakaHeartFlowChatting:
    runtime = object.__new__(MaisakaHeartFlowChatting)
    runtime._agent_state = runtime._STATE_RUNNING
    runtime._pending_wait_tool_call_id = None
    runtime._pending_wait_logical_turn_id = None
    runtime._pending_wait_started_at = None
    runtime._pending_wait_seconds = None
    runtime._wait_timeout_task = None
    runtime._proactive_trigger_message = None
    runtime._proactive_logical_turn_id = None
    runtime._internal_turn_queue = asyncio.Queue()
    runtime._mark_message_turn_unscheduled = lambda: None
    runtime._cancel_deferred_message_turn_task = lambda: None
    return runtime


def test_wait_pause_resume_completes_original_logical_turn() -> None:
    runtime = _build_runtime_stub()
    runtime._enter_wait_state(
        seconds=None,
        tool_call_id="wait-call-1",
        logical_turn_id="turn-wait-1",
    )
    runtime._pending_wait_started_at = time.time() - 1

    assert runtime._get_pending_wait_tool_call_ids() == {"wait-call-1"}
    runtime._enter_running_state()
    engine = MaisakaReasoningEngine(runtime)
    completed = engine._build_wait_completed_message(has_new_messages=True)

    assert completed.tool_call_id == "wait-call-1"
    assert completed.logical_turn_id == "turn-wait-1"
    assert completed.timestamp <= datetime.now()
    assert runtime._get_pending_wait_tool_call_ids() == set()


def test_wait_requires_registered_call_and_turn() -> None:
    runtime = _build_runtime_stub()
    with pytest.raises(ValueError, match="tool_call_id 和 logical_turn_id"):
        runtime._enter_wait_state(tool_call_id="wait-call-1", logical_turn_id=None)
    with pytest.raises(RuntimeError, match="不存在可补齐"):
        runtime._consume_pending_wait_state()


def test_proactive_switch_trigger_carries_source_logical_turn() -> None:
    runtime = _build_runtime_stub()
    runtime._agent_state = runtime._STATE_WAIT
    trigger = object()

    runtime._queue_proactive_turn(trigger, logical_turn_id="turn-switch-1")  # type: ignore[arg-type]

    assert runtime._agent_state == runtime._STATE_RUNNING
    assert runtime._consume_proactive_trigger_message() is trigger
    assert runtime._consume_proactive_logical_turn_id() == "turn-switch-1"
    assert runtime._internal_turn_queue.get_nowait() == "proactive"


@pytest.mark.asyncio
async def test_queued_proactive_turn_resumes_wait_entered_after_enqueue() -> None:
    runtime = _build_runtime_stub()
    trigger = object()
    # 主动任务先排队，当前 Planner 随后调用 wait，再消费已排队的主动任务。
    runtime._queue_proactive_turn(trigger, logical_turn_id="turn-proactive")  # type: ignore[arg-type]
    runtime._enter_wait_state(tool_call_id="wait-call", logical_turn_id="turn-wait")
    runtime._is_reply_frequency_silent = lambda: False
    runtime._has_pending_messages = lambda: False
    runtime._clear_focus_cooldown_wakeup_scheduled = lambda: None
    runtime.restore_proactive_user_context = AsyncMock()
    runtime._chat_history = []
    engine = MaisakaReasoningEngine(runtime)
    engine._emit_flow_step = AsyncMock()

    context = await engine._prepare_turn_start_context(runtime._internal_turn_queue.get_nowait())

    assert context.trigger_message is trigger
    assert context.logical_turn_id == "turn-wait"
    assert runtime._chat_history[0].tool_call_id == "wait-call"
    assert runtime._get_pending_wait_tool_call_ids() == set()
    assert runtime._agent_state == runtime._STATE_RUNNING


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [False, True])
async def test_late_self_echo_persists_without_waking_or_completing_wait(waiting: bool) -> None:
    runtime = _build_runtime_stub()
    if waiting:
        runtime._enter_wait_state(tool_call_id="wait-call", logical_turn_id="turn-wait")
    echo = SimpleNamespace(platform="qq", message_info=SimpleNamespace(user_info=SimpleNamespace(user_id="bot")))
    runtime.message_cache = [echo]
    runtime._last_processed_index = 0
    runtime._is_reply_frequency_silent = lambda: False
    runtime._has_pending_messages = lambda: runtime._last_processed_index < len(runtime.message_cache)

    def collect():
        pending = runtime.message_cache[runtime._last_processed_index:]
        runtime._last_processed_index = len(runtime.message_cache)
        return pending

    runtime._collect_pending_messages = collect
    engine = MaisakaReasoningEngine(runtime)
    engine._ingest_messages = AsyncMock()
    engine._pi_runner = SimpleNamespace(ingest=AsyncMock(), run=AsyncMock(), _schedule_context_maintenance=Mock())
    with patch("src.maisaka.reasoning_engine.is_bot_self", return_value=True):
        context = await engine._prepare_turn_start_context("message")
    assert context.trigger_message is None
    assert runtime._last_processed_index == 1
    engine._ingest_messages.assert_awaited_once_with([echo])
    engine._pi_runner.ingest.assert_awaited_once()
    engine._pi_runner.run.assert_not_called()
    engine._pi_runner._schedule_context_maintenance.assert_called_once()
    assert runtime._agent_state == (runtime._STATE_WAIT if waiting else runtime._STATE_RUNNING)
    assert runtime._get_pending_wait_tool_call_ids() == ({"wait-call"} if waiting else set())
