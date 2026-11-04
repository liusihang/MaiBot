from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import copy
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
        session_id="real-group", _monitor_run_id="run-a", _cycle_counter=0, _agent_state="running",
        _update_stage_status=Mock(),
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
        session_id="real-group", _monitor_run_id="run-a", _cycle_counter=0, _agent_state="running",
        _update_stage_status=Mock(),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_running_child_has_no_failure_result_and_updates_parent_stage(monkeypatch, failed):
    snapshots = []

    async def capture(**payload):
        snapshots.append(copy.deepcopy(payload))

    monkeypatch.setattr(observer, "emit_planner_snapshot", capture)
    runtime = SimpleNamespace(
        session_id="group", _monitor_run_id="run", _cycle_counter=0, _agent_state="running",
        _update_stage_status=Mock(),
    )
    monitor = observer.PiObserver(runtime)
    await monitor.handle({"type": "message_start", "message": {"role": "assistant"}})
    assert runtime._update_stage_status.call_args.args[0] == "思考中"
    await monitor.handle({
        "type": "message_end",
        "message": {"role": "assistant", "content": [
            {"type": "toolCall", "id": "child-call", "name": "subagent", "arguments": {"wait": True}},
        ]},
    })
    # The dashboard treats every entry in tools as a completed receipt. Pending
    # calls belong only in planner_tool_calls, so success=false cannot mean running.
    assert snapshots[-1]["tools"] == []
    assert snapshots[-1]["planner_tool_calls"][0]["id"] == "child-call"
    await monitor.handle({"type": "tool_execution_start", "toolCallId": "child-call"})
    assert snapshots[-1]["tools"] == []
    assert snapshots[-1]["active_tool_call_id"] == "child-call"
    assert runtime._update_stage_status.call_args.args[0] == "等待子代理"

    await monitor.handle({
        "type": "message_end", "message": {
            "role": "toolResult", "toolCallId": "child-call", "isError": failed,
            "content": [{"type": "text", "text": "child result"}],
        },
    })
    assert snapshots[-1]["tools"][0]["success"] is not failed
    assert snapshots[-1]["tools"][0]["ended_at"] >= snapshots[-1]["tools"][0]["started_at"]
    assert snapshots[-1]["active_tool_call_id"] == ""
    assert runtime._update_stage_status.call_args.args[0] == "思考中"
    assert all(snapshot["tools"] == [] for snapshot in snapshots[:-1])


@pytest.mark.asyncio
async def test_parallel_tool_receipt_keeps_other_execution_active(monkeypatch):
    emit = AsyncMock()
    monkeypatch.setattr(observer, "emit_planner_snapshot", emit)
    runtime = SimpleNamespace(
        session_id="group", _monitor_run_id="run", _cycle_counter=0, _agent_state="running",
        _update_stage_status=Mock(),
    )
    monitor = observer.PiObserver(runtime)
    await monitor.handle({"type": "message_start", "message": {"role": "assistant"}})
    await monitor.handle({"type": "message_end", "message": {
        "role": "assistant", "content": [
            {"type": "toolCall", "id": "a", "name": "subagent", "arguments": {"wait": True}},
            {"type": "toolCall", "id": "b", "name": "web_fetch", "arguments": {}},
        ],
    }})
    for call_id in ["a", "b"]:
        await monitor.handle({"type": "tool_execution_start", "toolCallId": call_id})
    await monitor.handle({"type": "message_end", "message": {
        "role": "toolResult", "toolCallId": "b", "content": [], "isError": False,
    }})
    assert emit.call_args.kwargs["active_tool_call_id"] == "a"
    assert [tool["tool_call_id"] for tool in emit.call_args.kwargs["tools"]] == ["b"]
    assert runtime._update_stage_status.call_args.args[0] == "等待子代理"


@pytest.mark.asyncio
@pytest.mark.parametrize("name,arguments,expected", [
    ("reply", {}, "发送回复"),
    ("web_fetch", {}, "工具执行"),
    ("subagent", {"wait": False}, "工具执行"),
    ("read_subagent", {"wait": True}, "等待子代理"),
])
async def test_stage_follows_actual_tool_activity(monkeypatch, name, arguments, expected):
    monkeypatch.setattr(observer, "emit_planner_snapshot", AsyncMock())
    runtime = SimpleNamespace(
        session_id="group", _monitor_run_id="run", _cycle_counter=0, _agent_state="running",
        _update_stage_status=Mock(),
    )
    monitor = observer.PiObserver(runtime)
    await monitor.handle({"type": "message_end", "message": {
        "role": "assistant", "content": [
            {"type": "toolCall", "id": "c", "name": name, "arguments": arguments},
        ],
    }})
    await monitor.handle({"type": "tool_execution_start", "toolCallId": "c"})
    assert runtime._update_stage_status.call_args.args[0] == expected
