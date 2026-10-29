from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.maisaka import runtime as runtime_module
from src.maisaka.reasoning_engine import MaisakaReasoningEngine
from src.maisaka.runtime import MaisakaHeartFlowChatting


@pytest.mark.asyncio
async def test_intake_is_durable_before_scheduling_or_cooldown(monkeypatch):
    monkeypatch.setattr(runtime_module.global_config.chat, "pi_agent_enabled", True)
    order = []
    message = SimpleNamespace(message_id="incoming", received_at=datetime.now(timezone.utc))
    runtime = SimpleNamespace(
        _focus_group=None,
        _running=True,
        _ensure_background_tasks_running=Mock(),
        _quote_counter=SimpleNamespace(record=Mock()),
        _last_message_received_at=0,
        _update_message_trigger_state=Mock(),
        _observe_message_debounce=Mock(),
        message_cache=[],
        _emit_monitor_message_ingested=Mock(),
        _prune_processed_message_cache=Mock(),
        _request_planner_interrupt_for_message=Mock(),
    )

    async def persist(msg):
        assert runtime.message_cache == []
        assert msg is message
        order.append("durable_inbox")

    runtime._reasoning_engine = SimpleNamespace(record_pi_pending=AsyncMock(side_effect=persist))
    runtime._schedule_message_turn = Mock(side_effect=lambda: order.append("schedule"))
    await MaisakaHeartFlowChatting._register_message(runtime, message)
    assert order == ["durable_inbox", "schedule"]
    assert runtime.message_cache == [message]


@pytest.mark.asyncio
async def test_restart_restores_inbox_by_exact_source_id_not_recent_window(monkeypatch):
    from src.common import message_repository

    source = SimpleNamespace(message_id="old-outside-recent-window")
    found = Mock(return_value=[source])
    monkeypatch.setattr(message_repository, "find_messages", found)
    runtime = SimpleNamespace(session_id="actual-group", message_cache=[], _last_processed_index=0)
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = runtime
    engine._pi_runner = SimpleNamespace(
        ensure_initialized=AsyncMock(),
        _db=AsyncMock(return_value=[SimpleNamespace(payload={"sourceMessageId": source.message_id})]),
    )
    await engine.restore_pi_pending()
    found.assert_called_once_with(session_id="actual-group", message_id=source.message_id, limit=1)
    assert runtime.message_cache == [source]
    assert runtime._last_processed_index == 0
