from unittest.mock import AsyncMock, Mock

import sqlite3

import pytest

from src.core.tooling import ToolExecutionResult, ToolSpec
from src.maisaka.pi_runtime import observer
from src.maisaka.pi_runtime.runner import PiGroupRunner
from pytests.maisaka.test_pi_harness_host import engine, history, wire_server


@pytest.mark.asyncio
async def test_tool_receipt_survives_auxiliary_database_record_failure(monkeypatch, tmp_path):
    """完成的 stop 不因后台工具记录写入失败而变为未知并锁死后续轮次。"""
    from src.maisaka.builtin_tool.stop import get_tool_spec

    monkeypatch.setattr(observer.PiObserver, "handle", AsyncMock())
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("stop", {}), ("stop", {})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        host._runtime._tool_registry.list_tools.return_value = [get_tool_spec()]

        async def invoke(invocation, context, **kwargs):
            invoked.append(invocation)
            return ToolExecutionResult("stop", True, content="Stopped", metadata={"pause_execution": True})

        host._runtime._tool_registry.invoke = invoke
        host._record_tool_execution_effects.side_effect = sqlite3.OperationalError("database is locked")
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            await runner._db("assert_no_unresolved_executions", runner.session_id)
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            assert receipt.status == "succeeded" and receipt.result["terminate"]
            await runner.run([])
            assert len(requests) == len(invoked) == 2
            assert (await runner.bridge.request("state", sessionId=runner.session_id))["unresolvedToolCalls"] == []
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_interrupted_builtin_stop_recovers_without_redispatch(monkeypatch, tmp_path):
    """stop 的回执写入中断后，重启仅核对本地控制，不重新执行或发送。"""
    from src.maisaka.builtin_tool.stop import get_tool_spec

    monkeypatch.setattr(observer.PiObserver, "handle", AsyncMock())
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("stop", {}), ("stop", {})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        host._runtime._tool_registry.list_tools.return_value = [get_tool_spec()]

        async def invoke(invocation, context, **kwargs):
            invoked.append(invocation)
            return ToolExecutionResult("stop", True, content="Stopped", metadata={"pause_execution": True})

        host._runtime._tool_registry.invoke = invoke
        first = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await first.ensure_initialized()
            first.ledger.record_tool_outcome = Mock(side_effect=sqlite3.OperationalError("database is locked"))
            with pytest.raises(RuntimeError, match="requires reconciliation"):
                await first.run([])
            assert (await first._db("lookup_tool_execution", first.session_id, "call-1")).status == "unknown"
        finally:
            await first.close()

        restored = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await restored.ensure_initialized()
            await restored._db("assert_no_unresolved_executions", restored.session_id)
            receipt = await restored._db("lookup_tool_execution", restored.session_id, "call-1")
            assert receipt.status == "failed" and receipt.result["isError"]
            assert (await restored.bridge.request("state", sessionId=restored.session_id))["unresolvedToolCalls"] == []
            assert len(invoked) == len(requests) == 1
            await restored.run([])
            assert len(invoked) == len(requests) == 2
        finally:
            await restored.close()


@pytest.mark.asyncio
async def test_default_child_can_search_and_read_using_the_real_sdk(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    monkeypatch.setattr(observer.PiObserver, "handle", AsyncMock())
    calls = [
        ("subagent", {"task": "Research and return sources"}),
        ("gpt_web_search", {"query": "topic"}),
        ("gpt_read_search_result", {"result_id": "source"}),
        {"text": "supported conclusion"},
        {"text": "parent finished"},
    ]
    with wire_server(calls) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        specs = await host._runtime._tool_registry.list_tools()
        for name in ("gpt_web_search", "gpt_read_search_result"):
            specs.append(
                ToolSpec(
                    name=name,
                    description="audited default search",
                    provider_name="plugin_runtime",
                    provider_type="plugin",
                    metadata={"plugin_id": "local.gpt-web-search"},
                    parameters_schema={"type": "object", "additionalProperties": True},
                )
            )
        host._runtime._chat_history.append(history("research", "m1"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert [item.tool_name for item in invoked] == ["gpt_web_search", "gpt_read_search_result"]
            child_tools = {item["function"]["name"] for item in requests[1]["tools"]}
            assert {"gpt_web_search", "gpt_read_search_result"} <= child_tools
            assert child_tools == {item["function"]["name"] for item in requests[0]["tools"]}
            assert "send_web_image" not in child_tools
            assert "supported conclusion" in str(requests[-1])
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_real_sdk_main_events_reach_existing_monitor_and_usage(monkeypatch, tmp_path):
    emit, account, cache = AsyncMock(), AsyncMock(), Mock()
    monkeypatch.setattr(observer, "emit_planner_snapshot", emit)
    monkeypatch.setattr(observer, "record_pi_usage", account)
    monkeypatch.setattr(observer, "record_llm_cache_usage", cache)
    monkeypatch.setattr(observer, "export_harness_config", lambda: {})
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("reply", {"text": "exact answer"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        runtime = host._runtime
        runtime._monitor_run_id, runtime._cycle_counter = "integration-run", 0
        runtime._chat_history.append(history("question", "m1"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert len(requests) == runtime._cycle_counter == account.await_count == 2
            assert cache.call_count == 2
            finalized = [
                call.kwargs for call in emit.call_args_list if call.kwargs["event_type"] == "planner.finalized"
            ]
            assert len(finalized) == 2
            assert finalized[0]["planner_model_name"] == "mock"
            assert finalized[0]["planner_prompt_tokens"] == 10
            assert finalized[0]["tools"][0]["tool_args"] == {"text": "exact answer"}
            assert finalized[0]["tools"][0]["success"] is True
            assert finalized[1]["planner_total_tokens"] == 12
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_known_reply_validation_failure_does_not_pause_the_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(observer.PiObserver, "handle", AsyncMock())
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("reply", {"text": "answer", "msg_id": "invalid"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        for spec in await host._runtime._tool_registry.list_tools():
            spec.provider_type = "builtin"

        async def reject(invocation, context, **kwargs):
            return ToolExecutionResult(
                "reply", False, error_message="target not found", metadata={"execution_outcome": "not_executed"}
            )

        host._runtime._tool_registry.invoke = reject
        host._runtime._chat_history.append(history("question", "m1"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert len(requests) == 2
            await runner._db("assert_no_unresolved_executions", runner.session_id)
            state = await runner.bridge.request("state", sessionId=runner.session_id)
            assert state["paused"] is False
            assert state["unresolvedToolCalls"] == []
        finally:
            await runner.close()
