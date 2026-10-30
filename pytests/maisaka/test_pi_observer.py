from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.maisaka.pi_runtime import observer


@pytest.mark.asyncio
async def test_public_round_tools_usage_and_cache_are_observed_once(monkeypatch):
    emit, account, cache = AsyncMock(), AsyncMock(), Mock()
    monkeypatch.setattr(observer, "emit_planner_snapshot", emit)
    monkeypatch.setattr(observer, "record_pi_usage", account)
    monkeypatch.setattr(observer, "record_llm_cache_usage", cache)
    monkeypatch.setattr(observer, "export_harness_config", lambda: {"models": []})
    runtime = SimpleNamespace(
        session_id="real-group", _monitor_run_id="run-a", _cycle_counter=0, _agent_state="running"
    )
    monitor = observer.PiObserver(runtime)
    route = {"modelName": "group-chat", "modelIdentifier": "terra", "apiProvider": "provider-a"}
    base = {"route": route, "contextMessageCount": 12, "toolCount": 8}
    await monitor.handle({**base, "type": "message_start", "message": {"role": "assistant", "content": []}})
    message = {
        "role": "assistant",
        "stopReason": "toolUse",
        "content": [{"type": "toolCall", "id": "c1", "name": "gpt_web_search", "arguments": {"query": "test"}}],
        "usage": {"input": 20, "cacheRead": 80, "cacheWrite": 0, "output": 5, "totalTokens": 105},
    }
    ended = {**base, "type": "message_end", "messageEntryId": "entry-1", "message": message}
    await monitor.handle(ended)
    await monitor.handle(ended)
    account.assert_awaited_once()
    assert account.call_args.kwargs["succeeded"] is True
    assert account.call_args.args[4] == "real-group"
    assert cache.call_args.kwargs["prompt_tokens"] == 100
    assert cache.call_args.kwargs["prompt_cache_hit_tokens"] == 80
    await monitor.handle({"type": "tool_execution_start", "toolCallId": "c1"})
    await monitor.handle(
        {
            "type": "message_end",
            "message": {
                "role": "toolResult",
                "toolCallId": "c1",
                "isError": False,
                "content": [{"type": "text", "text": "verified source"}],
            },
        }
    )
    await monitor.handle({"type": "turn_end"})
    final = emit.call_args.kwargs
    assert final["event_type"] == "planner.finalized"
    assert final["planner_selected_history_count"] == 12
    assert final["planner_prompt_tokens"] == 100
    assert final["planner_prompt_cache_hit_tokens"] == 80
    assert final["tools"][0]["success"] is True
    assert final["tools"][0]["tool_args"] == {"query": "test"}
    assert final["tools"][0]["summary"] == "verified source"
    assert runtime._cycle_counter == 1


@pytest.mark.asyncio
async def test_failed_model_and_observation_failure_do_not_execute_or_retry(monkeypatch):
    emit, account = AsyncMock(), AsyncMock()
    monkeypatch.setattr(observer, "emit_planner_snapshot", emit)
    monkeypatch.setattr(observer, "record_pi_usage", account)
    monkeypatch.setattr(observer, "record_llm_cache_usage", Mock())
    monkeypatch.setattr(observer, "export_harness_config", lambda: {})
    runtime = SimpleNamespace(
        session_id="real-group", _monitor_run_id="run-a", _cycle_counter=0, _agent_state="running"
    )
    monitor = observer.PiObserver(runtime)
    base = {"route": {"modelName": "group-chat"}, "contextMessageCount": 1, "toolCount": 1}
    await monitor.handle({**base, "type": "message_start", "message": {"role": "assistant"}})
    await monitor.handle(
        {
            **base,
            "type": "message_end",
            "messageEntryId": "failed",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "aborted",
                "usage": {"input": 2, "output": 0},
            },
        }
    )
    assert account.call_args.kwargs["succeeded"] is False
    await monitor.handle({"type": "turn_end"})
    assert emit.call_args.kwargs["planner_interrupted"] is True
    emit.side_effect = OSError("observer store unavailable")
    await monitor.handle({"type": "turn_end"})
    account.assert_awaited_once()


@pytest.mark.asyncio
async def test_child_usage_is_accounted_without_overwriting_parent_monitor(monkeypatch):
    emit, account, cache = AsyncMock(), AsyncMock(), Mock()
    monkeypatch.setattr(observer, "emit_planner_snapshot", emit)
    monkeypatch.setattr(observer, "record_pi_usage", account)
    monkeypatch.setattr(observer, "record_llm_cache_usage", cache)
    monkeypatch.setattr(observer, "export_harness_config", lambda: {})
    monitor = observer.PiObserver(
        SimpleNamespace(session_id="real-group", _cycle_counter=0), request_type="maisaka.sub_agent", publish=False
    )
    await monitor.handle({"type": "message_start", "message": {"role": "assistant"}})
    await monitor.handle(
        {
            "type": "message_end",
            "messageEntryId": "child-response",
            "route": {"modelName": "group-chat"},
            "message": {"role": "assistant", "content": [], "usage": {"input": 5, "output": 2}, "stopReason": "stop"},
        }
    )
    account.assert_awaited_once()
    assert account.call_args.kwargs["request_type"] == cache.call_args.kwargs["request_type"] == "maisaka.sub_agent"
    emit.assert_not_awaited()
