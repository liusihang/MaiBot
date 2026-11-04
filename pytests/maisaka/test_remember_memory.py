from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.core.tooling import ToolInvocation
from src.maisaka.builtin_tool import remember_memory
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext


@pytest.fixture
def setup_memory(monkeypatch):
    source = SimpleNamespace(
        session_id="memory-group", message_id="source-1", platform="qq",
        message_info=SimpleNamespace(
            user_info=SimpleNamespace(user_id="10001", user_nickname="OldNick", user_cardname="OldNick"),
            group_info=SimpleNamespace(group_id="test-group"),
        ),
        raw_message=MessageSequence([TextComponent("以后冷场时可以接这个梗。")]),
    )
    monkeypatch.setattr(remember_memory, "find_messages", lambda **kw: [source])
    monkeypatch.setattr(remember_memory, "is_bot_self", lambda *args: False)
    monkeypatch.setattr(remember_memory, "person_name_records", lambda **kw: {
        "person-a": {"platform": "qq", "user_id": "10001", "primary_name": "主称呼甲", "qq_name": "OldNick"},
    }, raising=False)
    service = SimpleNamespace(ingest_text=AsyncMock(return_value=SimpleNamespace(
        success=True, detail="", stored_ids=["paragraph-1"], skipped_ids=[],
    )), profile_admin=AsyncMock())
    monkeypatch.setattr(remember_memory, "memory_service", service)
    monkeypatch.setattr("src.maisaka.memory.heuristic_injector.heuristic_memory_injector.clear_session_reference", Mock())
    runtime = SimpleNamespace(
        session_id="memory-group", chat_stream=SimpleNamespace(group_id="test-group", user_id=""),
        find_source_message_by_id=lambda mid: None,
    )
    context = SimpleNamespace(runtime=runtime,
        build_success_result=BuiltinToolRuntimeContext.build_success_result,
        build_failure_result=BuiltinToolRuntimeContext.build_failure_result)
    invocation = ToolInvocation(tool_name="remember_memory", arguments={
        "content": "主称呼甲提议冷场时接一个梗。", "subject_name": "主称呼甲", "evidence_message_ids": ["source-1"],
    }, call_id="remember-call", session_id="memory-group", stream_id="memory-group")
    return source, service, context, invocation


@pytest.mark.asyncio
async def test_maintained_primary_name_resolves_to_source_user_id(setup_memory):
    source, service, context, invocation = setup_memory
    result = await remember_memory.handle_tool(context, invocation)
    assert result.success, result.error_message
    assert service.ingest_text.await_count == 1
    assert service.ingest_text.call_args.kwargs["person_ids"] == [remember_memory.get_person_id("qq", "10001")]


@pytest.mark.asyncio
async def test_name_validation_failure_is_known_not_executed(setup_memory):
    source, service, context, invocation = setup_memory
    invocation.arguments["subject_name"] = "不存在的人"
    result = await remember_memory.handle_tool(context, invocation)
    assert not result.success
    assert result.metadata["execution_outcome"] == "not_executed"
    service.ingest_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_evidence_is_known_not_executed(setup_memory, monkeypatch):
    source, service, context, invocation = setup_memory
    monkeypatch.setattr(remember_memory, "find_messages", lambda **kw: [])
    result = await remember_memory.handle_tool(context, invocation)
    assert not result.success
    assert result.metadata["execution_outcome"] == "not_executed"
    service.ingest_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_storage_failure_does_not_claim_write_was_not_executed(setup_memory):
    source, service, context, invocation = setup_memory
    service.ingest_text.return_value = SimpleNamespace(success=False, detail="storage_error", stored_ids=[], skipped_ids=[])
    result = await remember_memory.handle_tool(context, invocation)
    assert not result.success
    assert result.metadata.get("execution_outcome") != "not_executed"
    service.ingest_text.assert_awaited_once()
