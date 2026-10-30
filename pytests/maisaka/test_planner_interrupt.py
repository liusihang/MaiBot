from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import asyncio

import pytest

from src.maisaka import runtime as runtime_module
from src.maisaka.runtime import MaisakaHeartFlowChatting, PlannerInterruptController


def _runtime(monkeypatch, *, steering: bool) -> MaisakaHeartFlowChatting:
    monkeypatch.setattr(
        runtime_module,
        "global_config",
        SimpleNamespace(
            chat=SimpleNamespace(
                pi_agent_enabled=False, reply_timing=SimpleNamespace(planner_message_steering=steering)
            )
        ),
    )
    instance = object.__new__(MaisakaHeartFlowChatting)
    instance._agent_state = MaisakaHeartFlowChatting._STATE_RUNNING
    instance._planner_interrupt = PlannerInterruptController()
    instance.log_prefix = "[test]"
    instance.message_cache = []
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("steering", [False, True])
async def test_new_arrival_is_persisted_and_queued_without_cancelling_inflight_pi(monkeypatch, steering: bool) -> None:
    runtime = _runtime(monkeypatch, steering=steering)
    runtime._running = True
    runtime._focus_group = None
    runtime._last_message_received_at = 0.0
    runtime._ensure_background_tasks_running = Mock()
    runtime._quote_counter = SimpleNamespace(record=Mock())
    runtime._update_message_trigger_state = Mock()
    runtime._observe_message_debounce = Mock()
    runtime._emit_monitor_message_ingested = Mock()
    runtime._prune_processed_message_cache = Mock()
    runtime._schedule_message_turn = Mock()
    interrupt_flag = asyncio.Event()
    runtime._planner_interrupt.bind(interrupt_flag)
    release_model = asyncio.Event()
    inflight = asyncio.create_task(release_model.wait())
    source = SimpleNamespace(message_id="m1", received_at=datetime.now())
    order = []

    async def persist(message):
        assert message is source and runtime.message_cache == []
        assert not inflight.done() and not interrupt_flag.is_set()
        await asyncio.sleep(0)
        order.append("persisted")

    runtime._reasoning_engine = SimpleNamespace(record_pi_pending=AsyncMock(side_effect=persist))
    runtime._emit_monitor_message_ingested.side_effect = lambda message: order.append("queued")
    try:
        await runtime._register_message(source)
        assert order == ["persisted", "queued"]
        assert runtime.message_cache == [source]
        assert not interrupt_flag.is_set()
        assert not inflight.done() and not inflight.cancelled()
        runtime._reasoning_engine.record_pi_pending.assert_awaited_once_with(source)
        runtime._schedule_message_turn.assert_called_once()
    finally:
        release_model.set()
        await inflight


def test_same_planner_request_is_interrupted_once() -> None:
    controller = PlannerInterruptController()
    assert controller.request() == "idle"

    interrupt_flag = asyncio.Event()
    controller.bind(interrupt_flag)
    assert controller.request() == "request"
    assert controller.request() == "duplicate"

    controller.unbind(interrupt_flag)
    assert controller.request() == "idle"
