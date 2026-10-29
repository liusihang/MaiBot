from types import SimpleNamespace
from unittest.mock import AsyncMock
import json
import time

import pytest

from src.A_memorix.core.retrieval.dual_path import RetrievalResult
from src.A_memorix.core.retrieval.threshold import DynamicThresholdFilter
from src.A_memorix.core.runtime.services.search_hit_processing_service import MemorySearchHitProcessingService
from src.core.tooling import ToolExecutionContext, ToolInvocation
from src.maisaka.builtin_tool import query_memory, read_tool_result
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.maisaka.builtin_tool.query_memory import _normalize_optional_time
from src.maisaka.tool_result_store import ToolResultStore
from src.services.memory_service import MemoryHit, MemorySearchResult, MemoryService


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2026-09-01", "2026/09/01"),
        ("2026-09-01 12:30", "2026/09/01 12:30"),
        ("2026/09/01", "2026/09/01"),
        ("2026/09/01 12:30", "2026/09/01 12:30"),
    ],
)
def test_normalize_optional_time_accepts_slash_and_hyphen_formats(raw_value: str, expected: str) -> None:
    assert _normalize_optional_time(raw_value) == expected


def test_normalize_optional_time_does_not_relax_unsupported_formats() -> None:
    assert _normalize_optional_time("2026-09-01T12:30:00") == "2026-09-01T12:30:00"


@pytest.mark.asyncio
@pytest.mark.parametrize("resolved_id", ["target-person", ""])
async def test_explicit_person_uses_alias_mapping_instead_of_private_chat_id(monkeypatch, resolved_id):
    resolve = AsyncMock(return_value={"success": True, "person_id": resolved_id})
    monkeypatch.setattr(query_memory.memory_service, "profile_admin", resolve)
    result = await query_memory._resolve_person_id(
        person_name="另一人的旧昵称",
        platform="qq",
        user_id="current-chat-user",
        group_id="",
    )
    assert result == (resolved_id, "另一人的旧昵称")
    resolve.assert_awaited_once_with(action="resolve_identity", identifier="另一人的旧昵称")


@pytest.mark.asyncio
async def test_ambiguous_person_stops_query_instead_of_removing_person_filter(monkeypatch):
    monkeypatch.setattr(
        query_memory.memory_service, "profile_admin", AsyncMock(return_value={"success": True, "person_id": ""})
    )
    search = AsyncMock()
    monkeypatch.setattr(query_memory.memory_service, "search", search)
    runtime = SimpleNamespace(
        session_id="session", chat_stream=SimpleNamespace(platform="qq", user_id="current-chat-user", group_id="")
    )
    context = SimpleNamespace(runtime=runtime, build_failure_result=lambda name, error: {"error": error})
    invocation = ToolInvocation(tool_name="query_memory", arguments={"query": "旅行计划", "person_name": "重名用户"})
    result = await query_memory.handle_tool(context, invocation)
    assert "无法唯一确认" in result["error"]
    search.assert_not_awaited()


def _tool_context(session_id="chat-a", group_id="group-a"):
    return BuiltinToolRuntimeContext(
        SimpleNamespace(),
        SimpleNamespace(
            session_id=session_id,
            log_prefix="[memory-contract-test]",
            chat_stream=SimpleNamespace(platform="qq", user_id="user-a", group_id=group_id),
        ),
    )


def _invocation(**arguments):
    return ToolInvocation(tool_name="query_memory", arguments=arguments)


@pytest.fixture
def result_store(monkeypatch):
    store = ToolResultStore()
    monkeypatch.setattr(query_memory, "tool_result_store", store)
    monkeypatch.setattr(read_tool_result, "tool_result_store", store)
    return store


@pytest.mark.asyncio
async def test_person_and_time_miss_is_no_match_without_second_search(monkeypatch, result_store):
    monkeypatch.setattr(
        query_memory.memory_service, "profile_admin", AsyncMock(return_value={"success": True, "person_id": "person-a"})
    )
    search = AsyncMock(return_value=MemorySearchResult())
    monkeypatch.setattr(query_memory.memory_service, "search", search)
    result = await query_memory.handle_tool(
        _tool_context(),
        _invocation(query="旅行", person_name="甲", mode="hybrid", time_start="2026-09-01", time_end="2026-09-02"),
    )
    search.assert_awaited_once_with(
        "旅行",
        limit=5,
        mode="hybrid",
        chat_id="chat-a",
        person_id="person-a",
        time_start="2026/09/01",
        time_end="2026/09/02",
        respect_filter=True,
        user_id="user-a",
        group_id="group-a",
    )
    receipt = result.structured_content
    assert result.success and receipt["status"] == "no_match"
    assert receipt["hits"] == [] and not receipt["fallback_applied"]
    assert receipt["requested_scope"] == receipt["effective_scope"]
    assert receipt["effective_scope"]["person_id"] == "person-a"
    assert "replyer_memory_reference" not in result.metadata
    assert "memory_feedback_task" not in result.metadata
    assert "未自动扩大" in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["search", "aggregate"])
@pytest.mark.parametrize(("query", "effective_mode"), [("旅行", "hybrid"), ("", "time")])
async def test_supplied_time_cannot_be_silently_ignored(monkeypatch, result_store, mode, query, effective_mode):
    search = AsyncMock(return_value=MemorySearchResult())
    monkeypatch.setattr(query_memory.memory_service, "search", search)
    result = await query_memory.handle_tool(
        _tool_context(),
        _invocation(query=query, mode=mode, time_start="2026/09/01"),
    )
    assert search.await_count == 1
    assert search.await_args.kwargs["mode"] == effective_mode
    assert search.await_args.kwargs["time_start"] == "2026/09/01"
    assert result.structured_content["requested_scope"]["mode"] == mode
    assert result.structured_content["effective_scope"]["mode"] == effective_mode
    assert result.structured_content["time_start"] == "2026/09/01"


@pytest.mark.asyncio
@pytest.mark.parametrize(("success", "filtered", "status"), [(True, True, "filtered"), (False, False, "error")])
async def test_filtered_or_failed_payload_never_exposes_hits(monkeypatch, result_store, success, filtered, status):
    search = AsyncMock(
        return_value=MemorySearchResult(
            success=success,
            filtered=filtered,
            error="检索失败" if not success else "",
            hits=[MemoryHit(content="不可见的记忆", hash_value="secret")],
        )
    )
    monkeypatch.setattr(query_memory.memory_service, "search", search)
    result = await query_memory.handle_tool(_tool_context(), _invocation(query="旅行"))
    assert result.success is success
    assert result.structured_content["status"] == status
    assert result.structured_content["hits"] == []
    assert "replyer_memory_reference" not in result.metadata
    stored = result_store.get("chat-a", result.structured_content["result_id"])
    assert "不可见的记忆" not in stored.content
    assert "secret" not in stored.content


@pytest.mark.asyncio
async def test_large_result_is_bounded_and_raw_evidence_remains_session_scoped(monkeypatch, result_store):
    source = "chat_summary:chat-a"
    hits = [
        MemoryHit(
            content="长记忆" * 4000 + f"END_EVIDENCE_{index}",
            score=0.8,
            hit_type="paragraph",
            hash_value=f"paragraph-{index}",
            source="paragraph_search",
            metadata={"source": source, "person_ids": ["person-a"], "opaque_metadata": "原元数据" * 4000},
        )
        for index in range(20)
    ]
    monkeypatch.setattr(
        query_memory.memory_service,
        "search",
        AsyncMock(
            return_value=MemorySearchResult(
                summary="不能整段复制的汇总" * 1000,
                hits=hits,
            )
        ),
    )
    result = await query_memory.handle_tool(_tool_context(), _invocation(query="旅行", limit=20))
    receipt = result.structured_content
    assert result.success and receipt["status"] == "matched" and receipt["hit_count"] == 20
    assert len(result.content) <= query_memory.MEMORY_PREVIEW_MAX_CHARS
    assert result.get_history_content() == result.content.strip()
    assert len(result.metadata["replyer_memory_reference"]) <= query_memory.MEMORY_REFERENCE_MAX_CHARS
    assert len(json.dumps(receipt, ensure_ascii=False)) <= query_memory.MEMORY_STRUCTURED_MAX_CHARS
    assert receipt["preview_hit_count"] <= query_memory.MEMORY_PREVIEW_MAX_HITS
    assert receipt["truncated"] and result.metadata["bounded_preview"]
    assert result.metadata["reader_tool"] == "read_tool_result"
    assert source in result.content and "paragraph-0" in result.content
    raw = result_store.get("chat-a", receipt["result_id"])
    payload = json.loads(raw.content)
    assert payload["hits"] == [hit.to_dict() for hit in hits]
    assert "END_EVIDENCE_19" not in result.content
    assert "END_EVIDENCE_19" not in json.dumps(receipt, ensure_ascii=False)
    assert result_store.get("chat-b", receipt["result_id"]) is None
    read = await read_tool_result.handle_tool(
        _tool_context(),
        ToolInvocation(
            tool_name="read_tool_result",
            arguments={
                "result_id": receipt["result_id"],
                "keyword": "END_EVIDENCE_19",
            },
        ),
    )
    assert read.success and "END_EVIDENCE_19" in read.content
    cross_chat = await read_tool_result.handle_tool(
        _tool_context("chat-b", "group-b"),
        ToolInvocation(
            tool_name="read_tool_result",
            arguments={
                "result_id": receipt["result_id"],
            },
        ),
    )
    assert not cross_chat.success


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scores", "status", "evidence_count"), [([0.1, 0.05], "no_match", 0), ([0.9, 0.05], "matched", 1)]
)
async def test_real_threshold_backfill_survives_projection_and_is_not_explicit_evidence(
    monkeypatch,
    result_store,
    scores,
    status,
    evidence_count,
):
    candidates = [
        RetrievalResult(
            hash_value=f"paragraph-{index}",
            content=f"回忆 {index}",
            score=score,
            result_type="paragraph",
            source="paragraph_search",
            metadata={"source": "chat_summary:chat-a", "person_ids": ["person-a"]},
        )
        for index, score in enumerate(scores)
    ]
    retrieved = DynamicThresholdFilter().filter(candidates)
    assert len(retrieved) == len(candidates)  # 自动回忆的 min_results 召回下限不变。
    assert all("retrieval_evidence" not in item.metadata for item in candidates)
    rows = {item.hash_value: {"content": item.content, "metadata": dict(item.metadata)} for item in candidates}
    store = SimpleNamespace(
        get_paragraphs_by_hashes=lambda ids: {key: rows[key] for key in ids},
        project_fact_paragraph=lambda row: row,
    )
    hit_service = MemorySearchHitProcessingService(
        SimpleNamespace(
            metadata_store=store,
            _current_effective_filter_cache={"checked_at": time.time(), "needed": True},
        )
    )
    hits = [hit_service._retrieval_result_hit(item) for item in retrieved]
    hits = hit_service._filter_fact_paragraph_hits(hits)
    hits = hit_service._filter_current_effective_hits(hits)
    host_result = MemoryService._coerce_search_result({"hits": hits})
    assert all("retrieval_evidence" in item.metadata for item in host_result.hits)
    monkeypatch.setattr(query_memory.memory_service, "search", AsyncMock(return_value=host_result))
    result = await query_memory.handle_tool(_tool_context(), _invocation(query="不存在的具体往事"))
    assert result.structured_content["status"] == status
    assert result.structured_content["hit_count"] == evidence_count
    assert result.structured_content["candidate_only_count"] == len(scores) - evidence_count
    raw = json.loads(result_store.get("chat-a", result.structured_content["result_id"]).content)
    assert len(raw["hits"]) == evidence_count
    assert len(raw["excluded_candidates"]) == len(scores) - evidence_count
    assert all(item["reason"] == "relevance_threshold_not_met" for item in raw["excluded_candidates"])
    assert "不是回答依据" in raw["candidate_note"]
    assert "回忆 1" not in json.dumps(raw, ensure_ascii=False)
    read = await read_tool_result.handle_tool(
        _tool_context(),
        ToolInvocation(
            tool_name="read_tool_result",
            arguments={
                "result_id": result.structured_content["result_id"],
                "keyword": "回忆 1",
            },
        ),
    )
    assert read.success and "没有这个关键词" in read.content


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["time", "episode"])
async def test_unmarked_time_or_episode_hits_do_not_use_semantic_score_cutoff(monkeypatch, result_store, mode):
    monkeypatch.setattr(
        query_memory.memory_service,
        "search",
        AsyncMock(
            return_value=MemorySearchResult(
                hits=[
                    MemoryHit(content="在时间范围内的经历", score=0.0, hit_type=mode, source="chat_summary:chat-a"),
                ]
            )
        ),
    )
    result = await query_memory.handle_tool(_tool_context(), _invocation(mode=mode, time_start="2026/09/01"))
    assert result.structured_content["status"] == "matched"


@pytest.mark.asyncio
async def test_missing_real_session_cannot_fall_back_to_global_scope(monkeypatch):
    search = AsyncMock()
    monkeypatch.setattr(query_memory.memory_service, "search", search)
    result = await query_memory.handle_tool(_tool_context(session_id=""), _invocation(query="旅行"))
    assert not result.success and "真实聊天会话" in result.error_message
    search.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["invocation", "execution_context"])
async def test_mismatched_session_never_searches_another_chat(monkeypatch, mismatch):
    search = AsyncMock()
    monkeypatch.setattr(query_memory.memory_service, "search", search)
    invocation = ToolInvocation(
        tool_name="query_memory",
        arguments={"query": "旅行"},
        session_id="chat-b" if mismatch == "invocation" else "chat-a",
    )
    context = ToolExecutionContext(session_id="chat-b" if mismatch == "execution_context" else "chat-a")
    result = await query_memory.handle_tool(_tool_context(), invocation, context)
    assert not result.success and "不一致" in result.error_message
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_json_escaped_scope_and_metadata_still_fit_structured_budget(monkeypatch, result_store):
    escaped = "\x01" * 5000
    monkeypatch.setattr(
        query_memory.memory_service,
        "search",
        AsyncMock(
            return_value=MemorySearchResult(
                hits=[
                    MemoryHit(
                        content=escaped,
                        hash_value=escaped,
                        source=escaped,
                        metadata={
                            "source": escaped,
                            "person_ids": [escaped] * 4,
                            "evidence_message_ids": [escaped] * 4,
                        },
                    )
                    for _ in range(20)
                ]
            )
        ),
    )
    result = await query_memory.handle_tool(_tool_context(), _invocation(query=escaped, limit=20))
    assert len(json.dumps(result.structured_content, ensure_ascii=False)) <= query_memory.MEMORY_STRUCTURED_MAX_CHARS
    assert len(result.get_history_content()) <= query_memory.MEMORY_PREVIEW_MAX_CHARS


@pytest.mark.asyncio
async def test_long_provenance_ids_are_omitted_not_fabricated(monkeypatch, result_store):
    identifier = "opaque-" + "a" * 300
    source = "https://example.com/" + "a" * 300
    original = MemoryHit(
        content="有来源的记忆",
        hash_value=identifier,
        episode_id=identifier,
        source=source,
        metadata={"source": source, "person_ids": [identifier], "evidence_message_ids": [identifier]},
    )
    monkeypatch.setattr(
        query_memory.memory_service,
        "profile_admin",
        AsyncMock(
            return_value={
                "success": True,
                "person_id": identifier,
            }
        ),
    )
    monkeypatch.setattr(
        query_memory.memory_service, "search", AsyncMock(return_value=MemorySearchResult(hits=[original]))
    )
    result = await query_memory.handle_tool(_tool_context(), _invocation(query="旅行", person_name="甲"))
    receipt = result.structured_content
    hit = receipt["hits"][0]
    assert hit["hash"] == hit["episode_id"] == hit["source"] == ""
    assert hit["metadata"]["person_ids"] == hit["metadata"]["evidence_message_ids"] == []
    assert "metadata.source" in hit["omitted_fields"] and "source" not in hit["metadata"]
    assert "来源见原文" in result.content
    assert receipt["person_id"] is None and "person_id" in receipt["omitted_fields"]
    assert receipt["effective_scope"]["person_id"] is None
    assert "person_id" in receipt["effective_scope"]["omitted_fields"]
    raw = json.loads(result_store.get("chat-a", receipt["result_id"]).content)
    assert raw["hits"] == [original.to_dict()] and raw["effective_scope"]["person_id"] == identifier
