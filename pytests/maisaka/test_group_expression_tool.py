"""Offline contract checks for scoped plain-text group-language lookup."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core.tooling import ToolInvocation
from src.maisaka.builtin_tool import build_builtin_tool_handlers, get_all_builtin_tool_specs, query_group_expressions
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext


def runtime(session_id="host-session"):
    return SimpleNamespace(
        runtime=SimpleNamespace(session_id=session_id),
        build_success_result=BuiltinToolRuntimeContext.build_success_result,
        build_failure_result=BuiltinToolRuntimeContext.build_failure_result,
    )


@pytest.mark.asyncio
async def test_registered_tool_uses_host_scope_and_preserves_grounded_results(monkeypatch):
    value = {
        "candidates": [
            {
                "kind": "expression",
                "phrase": "起飞",
                "meaning": "进展很好",
                "situation": "庆祝成功",
                "examples": [{"text": "这把起飞了", "is_source": True}],
            }
        ],
        "coverage": {"mode": "hybrid", "complete": True},
    }
    search = AsyncMock(return_value=value)
    monkeypatch.setattr(query_group_expressions, "_search", search)
    specs = {spec.name: spec for spec in get_all_builtin_tool_specs()}
    assert specs["query_group_expressions"].metadata["visibility"] == "visible"
    safety = specs["query_group_expressions"].execution_contract()["safety"]
    assert safety["read_only"] is True and safety["destructive"] is False
    assert "session_id" not in specs["query_group_expressions"].parameters_schema["properties"]
    handler = build_builtin_tool_handlers(runtime())["query_group_expressions"]
    result = await handler(
        ToolInvocation(
            tool_name="query_group_expressions", arguments={"intent": " 祝贺通关 ", "context": "朋友玩游戏", "limit": 3}
        )
    )
    search.assert_awaited_once_with("host-session", "祝贺通关", "朋友玩游戏", 3)
    assert result.success
    assert result.get_history_content() == "表达风格：起飞\n情景：庆祝成功\n例句：这把起飞了"
    assert result.structured_content["candidates"] == value["candidates"]
    assert result.structured_content["coverage"] == value["coverage"]
    assert result.structured_content["authority"].startswith("untrusted_")
    assert not result.post_history_messages  # normal receipt, not another injected reference


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"intent": " "},
        {"intent": "x" * 1001},
        {"intent": 1},
        {"intent": "ok", "context": "x" * 2001},
        {"intent": "ok", "context": None},
        {"intent": "ok", "limit": True},
        {"intent": "ok", "limit": 0},
        {"intent": "ok", "limit": 7},
        {"intent": "ok", "limit": 1.5},
        {"intent": "ok", "session_id": "another-group"},
    ],
)
async def test_invalid_or_cross_scope_arguments_never_search(monkeypatch, arguments):
    search = AsyncMock()
    monkeypatch.setattr(query_group_expressions, "_search", search)
    result = await query_group_expressions.handle_tool(
        runtime(), ToolInvocation(tool_name="query_group_expressions", arguments=arguments)
    )
    assert not result.success
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_host_session_never_searches(monkeypatch):
    search = AsyncMock()
    monkeypatch.setattr(query_group_expressions, "_search", search)
    result = await query_group_expressions.handle_tool(
        runtime(""), ToolInvocation(tool_name="query_group_expressions", arguments={"intent": "ok"})
    )
    assert not result.success
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_or_partial_results_do_not_trigger_selector_or_random_fallback(monkeypatch):
    search = AsyncMock(return_value={"candidates": [], "coverage": {"mode": "lexical_only", "complete": False}})
    monkeypatch.setattr(query_group_expressions, "_search", search)
    result = await query_group_expressions.handle_tool(
        runtime(), ToolInvocation(tool_name="query_group_expressions", arguments={"intent": "ok"})
    )
    assert result.success and result.structured_content["candidates"] == []
    assert result.structured_content["coverage"]["complete"] is False
    search.assert_awaited_once_with("host-session", "ok", "", 6)


@pytest.mark.asyncio
async def test_failure_is_explicit_and_does_not_leak_exception_details(monkeypatch):
    monkeypatch.setattr(
        query_group_expressions, "_search", AsyncMock(side_effect=RuntimeError("SECRET other group data"))
    )
    result = await query_group_expressions.handle_tool(
        runtime(), ToolInvocation(tool_name="query_group_expressions", arguments={"intent": "ok"})
    )
    assert not result.success and result.structured_content == {"error": "RuntimeError"}
    assert "SECRET" not in result.error_message


@pytest.mark.parametrize("language", ["zh-CN", "en-US", "ja-JP"])
def test_active_prompts_explain_intent_search_and_match_deployed_version(language):
    relative = Path(language) / "maisaka_chat.prompt"
    source = (Path("prompts") / relative).read_text()
    deployed = (Path("deploy/qqbot/custom_prompts") / relative).read_text()
    assert source == deployed
    assert "query_group_expressions(intent, context)" in source
    assert "BM25 + embedding" in source and "RRF" in source


@pytest.mark.asyncio
async def test_mixed_six_candidates_are_plain_text_without_internal_receipt(monkeypatch):
    candidates = []
    for i in range(6):
        candidates.append(
            {
                "kind": "jargon" if i % 2 else "expression",
                "phrase": str(i),
                "situation": "庆祝成功",
                "meaning": "进展很好",
                "version": "HIDDEN_HASH",
                "ranking": {"rrf_score": 0.02},
                "examples": [
                    {"text": "真实例句", "is_source": True},
                    {"text": "NEIGHBOUR_NOT_USAGE", "is_source": False},
                ],
            }
        )
    search = AsyncMock(return_value={"candidates": candidates, "coverage": {"mode": "partial_hybrid"}})
    monkeypatch.setattr(query_group_expressions, "_search", search)
    result = await query_group_expressions.handle_tool(
        runtime(), ToolInvocation(tool_name="query_group_expressions", arguments={"intent": "庆祝"})
    )
    search.assert_awaited_once_with("host-session", "庆祝", "", 6)
    text = result.get_history_content()
    assert text.count("表达风格：") == text.count("黑话：") == 3
    assert text.count("例句：真实例句") == 6
    for diagnostic in ["coverage", "HIDDEN_HASH", "rrf_score", "以下仅为", "NEIGHBOUR_NOT_USAGE", "authority"]:
        assert diagnostic not in text
    assert result.structured_content["coverage"]["mode"] == "partial_hybrid"
    assert result.structured_content["candidates"] == candidates


def test_missing_examples_are_shown_without_inventing_text():
    assert (
        query_group_expressions._render_candidates(
            [{"kind": "jargon", "phrase": "上岸", "meaning": "达成目标", "examples": []}]
        )
        == "黑话：上岸\n含义：达成目标\n例句：暂无例句"
    )
    assert query_group_expressions._render_candidates([]) == "暂无匹配表达或黑话。"
