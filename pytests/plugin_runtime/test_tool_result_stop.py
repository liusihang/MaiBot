"""插件工具结束 Planner 返回协议测试。"""

import pytest

from src.plugin_runtime.component_query import ComponentQueryService
from src.plugin_runtime.host.component_registry import ComponentTypes, ToolEntry


def _build_tool_entry() -> ToolEntry:
    """构造最小插件工具条目。"""

    return ToolEntry(
        name="test_tool",
        component_type=ComponentTypes.TOOL.value,
        plugin_id="test_plugin",
        metadata={},
    )


@pytest.mark.parametrize("stop_after_execution", [True, False])
def test_parse_tool_result_maps_stop_after_execution(stop_after_execution: bool) -> None:
    result = ComponentQueryService._parse_tool_invoke_result(
        _build_tool_entry(),
        {
            "success": True,
            "message": "执行完成",
            "stop_after_execution": stop_after_execution,
        },
    )

    assert result.success is True
    assert result.stop_after_execution is stop_after_execution
    assert result.structured_content["stop_after_execution"] is stop_after_execution


def test_parse_tool_result_defaults_to_continue() -> None:
    result = ComponentQueryService._parse_tool_invoke_result(
        _build_tool_entry(),
        {"success": True, "message": "执行完成"},
    )

    assert result.success is True
    assert result.stop_after_execution is False


def test_parse_tool_result_rejects_non_boolean_stop_flag() -> None:
    result = ComponentQueryService._parse_tool_invoke_result(
        _build_tool_entry(),
        {"success": True, "stop_after_execution": "true"},
    )

    assert result.success is False
    assert result.stop_after_execution is False
    assert "必须为布尔值" in result.error_message


def test_model_arguments_cannot_override_trusted_tool_session() -> None:
    from src.core.tooling import ToolExecutionContext, ToolInvocation

    invocation = ToolInvocation(
        tool_name="test_tool",
        arguments={
            "stream_id": "forged",
            "chat_id": "forged",
            "session_id": "forged",
            "group_id": "forged",
            "user_id": "forged",
            "platform": "forged",
            "query": "evidence",
        },
    )
    context = ToolExecutionContext(session_id="real-session", group_id="real-group", platform="qq")
    result = ComponentQueryService._build_tool_invocation_payload(_build_tool_entry(), invocation, context)
    assert result == {
        "stream_id": "real-session",
        "chat_id": "real-session",
        "group_id": "real-group",
        "platform": "qq",
        "query": "evidence",
    }
    missing = ComponentQueryService._build_tool_invocation_payload(_build_tool_entry(), invocation, None)
    assert missing == {"query": "evidence"}


def test_declared_user_group_and_platform_arguments_remain_business_parameters() -> None:
    from src.core.tooling import ToolExecutionContext, ToolInvocation

    entry = ToolEntry(
        name="lookup_person",
        component_type=ComponentTypes.TOOL.value,
        plugin_id="test_plugin",
        metadata={
            "parameters_raw": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in ("user_id", "group_id", "platform", "stream_id")},
            }
        },
    )
    invocation = ToolInvocation(
        tool_name="lookup_person",
        arguments={
            "user_id": "target-user",
            "group_id": "target-group",
            "platform": "target-platform",
            "stream_id": "forged-session",
        },
    )
    context = ToolExecutionContext(session_id="real-session", user_id="caller", group_id="caller-group", platform="qq")
    result = ComponentQueryService._build_tool_invocation_payload(entry, invocation, context)
    assert result == {
        "user_id": "target-user",
        "group_id": "target-group",
        "platform": "target-platform",
        "stream_id": "real-session",
        "chat_id": "real-session",
    }
