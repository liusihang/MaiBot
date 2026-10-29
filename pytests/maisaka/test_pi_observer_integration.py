from unittest.mock import AsyncMock, Mock

import pytest

from src.core.tooling import ToolSpec
from src.maisaka.pi_runtime import observer
from src.maisaka.pi_runtime.runner import PiGroupRunner
from pytests.maisaka.test_pi_harness_host import engine, history, wire_server


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
            assert "reply" not in child_tools
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
