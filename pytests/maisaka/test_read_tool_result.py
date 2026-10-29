from types import SimpleNamespace
from typing import Any

import pytest

from src.config.config import global_config, model_config
from src.core.tooling import ToolInvocation
from src.llm_models.payload_content.tool_option import ToolCall
from src.maisaka.builtin_tool import get_all_builtin_tool_specs, read_tool_result
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.maisaka.tool_result_condenser import condense_tool_result, should_condense_tool_result
from src.maisaka.tool_result_store import MAX_STORED_RESULTS_PER_SESSION, ToolResultStore, tool_result_store
from src.web_search.config import WebSearchTaskConfig


def _tool_ctx(session_id: str) -> BuiltinToolRuntimeContext:
    return BuiltinToolRuntimeContext(SimpleNamespace(), SimpleNamespace(session_id=session_id))


def _invocation(**arguments: Any) -> ToolInvocation:
    return ToolInvocation(tool_name="read_tool_result", arguments=arguments, call_id="call-read")


def test_store_assigns_unique_scoped_handles_and_evicts_oldest() -> None:
    store = ToolResultStore()
    first = store.save("chat-a", "tool", "一")
    second = store.save("chat-a", "tool", "二")
    elsewhere = store.save("chat-b", "tool", "别的聊天")
    assert len({first, second, elsewhere}) == 3
    assert store.get("chat-a", second).content == "二"
    assert store.get("chat-b", second) is None
    assert store.get("chat-a", elsewhere) is None
    for index in range(MAX_STORED_RESULTS_PER_SESSION):
        last = store.save("chat-a", "tool", f"新结果{index}")
    assert store.get("chat-a", first) is None
    assert store.get("chat-a", last) is not None
    restarted = ToolResultStore()
    assert restarted.save("chat-a", "tool", "重启后新原文") != first
    assert restarted.get("chat-a", first) is None


def test_store_refuses_missing_session() -> None:
    with pytest.raises(ValueError, match="session_id"):
        ToolResultStore().save("", "tool", "raw")


@pytest.mark.asyncio
async def test_condensed_result_tells_planner_how_to_fetch_the_original() -> None:
    original = "插件A 仓库 https://example.com/a\n" * 200

    async def run_sub_agent(**kwargs: Any) -> Any:
        del kwargs
        return SimpleNamespace(content="共 200 个插件，代表：插件A。")

    runtime = SimpleNamespace(session_id="chat-condense", log_prefix="[test]", run_sub_agent=run_sub_agent)
    tool_call = ToolCall(call_id="call-1", func_name="maibot_plugin_list", args={})

    history_content = await condense_tool_result(runtime, tool_call, original, "想推荐插件")

    result_id = history_content.split("原始结果已保存为 ")[1].split("。")[0]
    assert history_content.splitlines() == [
        f'<压缩后信息；原始结果已保存为 {result_id}。需要上面没有的细节时，调用 read_tool_result(result_id="{result_id}") 分段读取原文，或加 keyword 参数直接查找。>',
        "共 200 个插件，代表：插件A。",
    ]
    assert tool_result_store.get("chat-condense", result_id).content == original


@pytest.mark.asyncio
async def test_read_tool_result_pages_through_original_by_offset() -> None:
    result_id = tool_result_store.save("chat-read", "maibot_release_info", "甲" * 2000 + "乙" * 1500)
    tool_ctx = _tool_ctx("chat-read")

    first = await read_tool_result.handle_tool(tool_ctx, _invocation(result_id=result_id))
    second = await read_tool_result.handle_tool(
        tool_ctx, _invocation(result_id=result_id, offset=2000, max_chars=99999)
    )

    assert first.success
    assert f"[{result_id} · maibot_release_info 原始结果 · 共 3500 字 · 本段 0–2000]" in first.content
    assert "后面还有 1500 字，用 offset=2000 继续读取" in first.content
    assert "乙" * 1500 in second.content
    assert "已到原文末尾" in second.content
    # 读回来的内容不会再触发整理
    assert not should_condense_tool_result("read_tool_result", "x" * 99999, success=True)


@pytest.mark.asyncio
async def test_read_tool_result_finds_keyword_snippets() -> None:
    content = "无关内容。" * 300 + "Pixiv 搜图插件 仓库 https://example.com/pixiv" + "无关内容。" * 300
    result_id = tool_result_store.save("chat-keyword", "maibot_plugin_list", content)
    tool_ctx = _tool_ctx("chat-keyword")

    found = await read_tool_result.handle_tool(tool_ctx, _invocation(result_id=result_id, keyword="pixiv"))
    missing = await read_tool_result.handle_tool(tool_ctx, _invocation(result_id=result_id, keyword="不存在的词"))

    assert "命中 2 处" in found.content
    assert "https://example.com/pixiv" in found.content
    assert len(found.content) < 1200
    assert "原文中没有这个关键词" in missing.content


@pytest.mark.asyncio
async def test_read_tool_result_reports_expired_ids() -> None:
    result = await read_tool_result.handle_tool(_tool_ctx("chat-empty"), _invocation(result_id="r99"))

    assert not result.success
    assert "重新调用原来的工具" in result.error_message


def test_read_tool_result_remains_exposed_without_condense_or_native_web(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_config.model_task_config, "web_search", WebSearchTaskConfig())
    monkeypatch.setattr(global_config.capabilities, "tool_result_condense", False)
    assert "read_tool_result" in {tool_spec.name for tool_spec in get_all_builtin_tool_specs()}

    monkeypatch.setattr(global_config.capabilities, "tool_result_condense", True)
    assert "read_tool_result" in {tool_spec.name for tool_spec in get_all_builtin_tool_specs()}


def test_web_results_can_be_read_without_condense(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_config.capabilities, "tool_result_condense", False)
    monkeypatch.setattr(model_config.model_task_config, "web_search", WebSearchTaskConfig(model_list=["step"]))
    assert "read_tool_result" in {tool_spec.name for tool_spec in get_all_builtin_tool_specs()}


def test_web_preview_skips_condense_but_raw_mcp_result_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_config.capabilities, "tool_result_condense_threshold", 3000)
    assert not should_condense_tool_result("web_search", "x" * 5000, success=True, already_previewed=True)
    assert should_condense_tool_result("web_search", "x" * 5000, success=True)


@pytest.mark.asyncio
async def test_keyword_reads_share_one_budget_and_can_continue() -> None:
    import re

    content = ("😀关键词" + "甲" * 600) * 20
    result_id = tool_result_store.save("chat-budget", "query_memory", content)
    first = await read_tool_result.handle_tool(
        _tool_ctx("chat-budget"), _invocation(result_id=result_id, keyword="关键词", max_chars=120)
    )
    ranges = [(int(a), int(b)) for a, b in re.findall(r"--- 位置 (\d+)–(\d+) ---", first.content)]
    assert sum(end - start for start, end in ranges) <= 120
    assert len(first.content) <= read_tool_result.READ_OUTPUT_MAX_CHARS
    assert len(ranges) == 1
    end = ranges[-1][1]
    second = await read_tool_result.handle_tool(
        _tool_ctx("chat-budget"), _invocation(result_id=result_id, keyword="关键词", offset=end, max_chars=120)
    )
    second_ranges = [(int(a), int(b)) for a, b in re.findall(r"--- 位置 (\d+)–(\d+) ---", second.content)]
    assert second_ranges[0][0] >= end
    for start, stop in [*ranges, *second_ranges]:
        assert content[start:stop] in first.content + second.content


@pytest.mark.asyncio
async def test_unicode_offsets_dense_matches_and_invalid_keyword_stay_bounded() -> None:
    import re

    content = "İ" * 20 + ("😀中文" * 5000)
    result_id = tool_result_store.save("chat-unicode", "a" * 100_000, content)
    ctx = _tool_ctx("chat-unicode")
    dense = await read_tool_result.handle_tool(ctx, _invocation(result_id=result_id, keyword="😀", max_chars=999999))
    assert dense.success and len(dense.content) <= read_tool_result.READ_OUTPUT_MAX_CHARS
    for start, end in re.findall(r"--- 位置 (\d+)–(\d+) ---", dense.content):
        assert content[int(start) : int(end)] in dense.content
    invalid = await read_tool_result.handle_tool(ctx, _invocation(result_id=result_id, keyword="中" * 100_000))
    assert not invalid.success and len(invalid.error_message) < 200


@pytest.mark.asyncio
async def test_expired_or_cross_session_handle_never_returns_other_evidence() -> None:
    from src.core.tooling import ToolExecutionContext

    result_id = tool_result_store.save("chat-original", "query_memory", "PRIVATE-ORIGINAL")
    tool_result_store.save("chat-other", "query_memory", "OTHER-FACT")
    result = await read_tool_result.handle_tool(_tool_ctx("chat-other"), _invocation(result_id=result_id))
    assert not result.success and "PRIVATE-ORIGINAL" not in result.content and "OTHER-FACT" not in result.content
    mismatch = await read_tool_result.handle_tool(
        _tool_ctx("chat-original"), _invocation(result_id=result_id), ToolExecutionContext(session_id="other")
    )
    assert not mismatch.success and "不一致" in mismatch.error_message
    for _ in range(MAX_STORED_RESULTS_PER_SESSION):
        tool_result_store.save("chat-original", "query_memory", "new")
    expired = await read_tool_result.handle_tool(_tool_ctx("chat-original"), _invocation(result_id=result_id))
    assert not expired.success and "过期" in expired.error_message
