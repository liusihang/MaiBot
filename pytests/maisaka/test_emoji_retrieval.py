from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.common.data_models.embedding_service_data_models import EmbeddingResult
from src.services import emoji_retrieval as retrieval


def emoji(tmp_path, name, description):
    path = tmp_path / name
    path.write_bytes(b"image")
    return SimpleNamespace(file_hash=name, description=description, full_path=path)


def result(vector, model="test"):
    return EmbeddingResult(embedding=vector, model_name=model, model_identifier=model, api_provider="test")


@pytest.mark.asyncio
async def test_ranks_top_three_caches_and_excludes_missing_or_unlabelled(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "_space", lambda value: value.model_name)
    values = {"happy": [1, 0], "a": [1, 0], "b": [0.9, 0.1], "c": [0.8, 0.2], "d": [0.7, 0.3]}
    client = SimpleNamespace(embed_text=AsyncMock(side_effect=lambda text, **_: result(values[text])))
    emojis = [emoji(tmp_path, name, name) for name in "abcd"]
    emojis += [emoji(tmp_path, "empty", "")]
    missing = emoji(tmp_path, "missing", "missing")
    missing.full_path.unlink()
    emojis += [missing, emojis[0]]
    index = retrieval.EmojiDescriptionIndex(tmp_path / "vectors.sqlite")
    hits = await retrieval.query_registered_emojis("happy", emojis, session_id="s", client=client, index=index)
    assert [item.file_hash for item, _ in hits.hits] == list("abc")
    assert client.embed_text.await_count == 5
    await retrieval.query_registered_emojis(
        "happy",
        emojis,
        session_id="s",
        client=client,
        index=retrieval.EmojiDescriptionIndex(tmp_path / "vectors.sqlite"),
    )
    assert client.embed_text.await_count == 6  # only the new query, never re-encode unchanged descriptions


@pytest.mark.asyncio
async def test_description_edits_and_vector_spaces_are_not_reused(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "_space", lambda value: value.model_name)
    client = SimpleNamespace(embed_text=AsyncMock(return_value=result([1, 0])))
    item = emoji(tmp_path, "a", "old")
    index = retrieval.EmojiDescriptionIndex(tmp_path / "vectors.sqlite")
    await retrieval.query_registered_emojis("q", [item], session_id="s", client=client, index=index)
    item.description = "new"
    await retrieval.query_registered_emojis("q", [item], session_id="s", client=client, index=index)
    assert client.embed_text.await_count == 4
    client.embed_text = AsyncMock(side_effect=[result([1, 0], "new-space"), result([1, 0], "wrong-space")])
    with pytest.raises(ValueError, match="different vector spaces"):
        await retrieval.query_registered_emojis("q", [item], session_id="s", client=client, index=index)


@pytest.mark.parametrize("vector", [[], [0, 0], [float("nan")], [float("inf")]])
def test_invalid_vectors_fail(vector):
    with pytest.raises(ValueError):
        retrieval.unit_vector(vector)


@pytest.mark.asyncio
async def test_query_allocates_only_returned_candidates(monkeypatch, tmp_path):
    from src.core.tooling import ToolInvocation
    from src.maisaka.builtin_tool import query_emoji

    item = emoji(tmp_path, "a", "delighted")
    monkeypatch.setattr(
        query_emoji,
        "query_registered_emojis",
        AsyncMock(return_value=retrieval.EmojiRetrievalResult([(item, 0.8)], 1, 1)),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(session_id="s"),
        allocate_emoji_candidates=AsyncMock(return_value={1000000000: "a"}),
        build_success_result=lambda *args, **kw: kw,
        build_failure_result=lambda *args, **kw: pytest.fail(str(args)),
    )
    value = await query_emoji.handle_tool(ctx, ToolInvocation(tool_name="query_emoji", arguments={"query": "happy"}))
    assert value["structured_content"]["candidates"][0]["candidate_index"] == 1000000000
    ctx.allocate_emoji_candidates.assert_awaited_once_with(["a"])


@pytest.mark.asyncio
async def test_send_requires_explicit_issued_candidate_and_never_selects(monkeypatch):
    from src.core.tooling import ToolInvocation
    from src.maisaka.builtin_tool import send_emoji

    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            _chat_history=[], log_prefix="test", session_id="s", chat_stream=SimpleNamespace(platform="qq")
        ),
        resolve_emoji_candidate_hash=AsyncMock(side_effect=ValueError("unknown candidate")),
        build_failure_result=lambda *args, **kw: {"success": False},
        build_success_result=lambda *args, **kw: pytest.fail("unexpected send"),
    )
    sender = AsyncMock()
    monkeypatch.setattr(send_emoji, "send_emoji_for_maisaka", sender)
    value = await send_emoji.handle_tool(ctx, ToolInvocation(tool_name="send_emoji", arguments={}))
    assert value["success"] is False
    sender.assert_not_called()


@pytest.mark.asyncio
async def test_empty_library_never_calls_embedding(tmp_path):
    client = SimpleNamespace(embed_text=AsyncMock())
    hits = await retrieval.query_registered_emojis("happy", [], session_id="s", client=client)
    assert hits.hits == []
    client.embed_text.assert_not_called()


@pytest.mark.asyncio
async def test_nonpositive_matches_are_not_randomly_replaced(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "_space", lambda value: value.model_name)
    client = SimpleNamespace(embed_text=AsyncMock(side_effect=[result([1, 0]), result([-1, 0])]))
    hits = await retrieval.query_registered_emojis(
        "q",
        [emoji(tmp_path, "a", "sad")],
        session_id="s",
        client=client,
        index=retrieval.EmojiDescriptionIndex(tmp_path / "i"),
    )
    assert hits.hits == []


@pytest.mark.asyncio
async def test_explicit_send_passes_selected_hash_and_preserves_target(monkeypatch, tmp_path):
    from src.core.tooling import ToolInvocation
    from src.emoji_system.maisaka_tool import MaisakaEmojiSendResult
    from src.maisaka.builtin_tool import send_emoji

    selected = emoji(tmp_path, "hash-a", "happy")
    target = object()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            _chat_history=[],
            log_prefix="test",
            session_id="s",
            chat_stream=SimpleNamespace(platform="qq"),
            find_source_message_by_id=lambda _: target,
        ),
        resolve_emoji_candidate_hash=AsyncMock(return_value="hash-a"),
        build_failure_result=lambda *args, **kw: pytest.fail(str(args)),
        build_success_result=lambda *args, **kw: kw,
    )
    monkeypatch.setattr(
        send_emoji.emoji_manager, "get_emoji_by_hash", lambda value: selected if value == "hash-a" else None
    )

    async def send(**kwargs):
        assert kwargs["expected_emoji_hash"] == "hash-a"
        assert kwargs["reply_message"] is target
        assert await kwargs["emoji_selector"]("", "", [], 64) == (selected, "")
        return MaisakaEmojiSendResult(True, "sent")

    sender = AsyncMock(side_effect=send)
    monkeypatch.setattr(send_emoji, "send_emoji_for_maisaka", sender)
    await send_emoji.handle_tool(
        ctx, ToolInvocation(tool_name="send_emoji", arguments={"candidate_index": 1000000000, "msg_id": "m"})
    )
    sender.assert_awaited_once()


def test_semantic_tools_replace_random_catalog_under_both_legacy_settings(monkeypatch):
    from src.maisaka import builtin_tool
    from src.maisaka.builtin_tool import reply
    from src.config import config as config_module

    for enabled in (False, True):
        monkeypatch.setattr(config_module.global_config.emoji, "use_new_send_logic", enabled)
        entries = {entry.name: entry for entry in builtin_tool.BUILTIN_TOOL_ENTRIES}
        assert builtin_tool._is_builtin_tool_enabled_by_config(entries["query_emoji"])
        assert builtin_tool._is_builtin_tool_enabled_by_config(entries["send_emoji"])
        assert not builtin_tool._is_builtin_tool_enabled_by_config(entries["show_emoji_list"])
        assert "attach_emoji" in reply.get_tool_spec().parameters_schema["properties"]


@pytest.mark.asyncio
async def test_hook_cannot_substitute_main_agent_choice(monkeypatch, tmp_path):
    from src.emoji_system import maisaka_tool

    chosen = emoji(tmp_path, "chosen", "happy")
    replacement = emoji(tmp_path, "other", "angry")

    async def hook(name, **kwargs):
        if name.endswith("after_select"):
            kwargs["selected_emoji_hash"] = "other"
        return SimpleNamespace(aborted=False, kwargs=kwargs)

    monkeypatch.setattr(maisaka_tool, "_get_runtime_manager", lambda: SimpleNamespace(invoke_hook=hook))
    monkeypatch.setattr(maisaka_tool, "_serialize_emoji_for_hook", lambda value: {"file_hash": value.file_hash})
    monkeypatch.setattr(maisaka_tool, "_resolve_selected_emoji", lambda value: replacement)
    value = await maisaka_tool.send_emoji_for_maisaka(
        stream_id="s",
        emoji_selector=AsyncMock(return_value=(chosen, "")),
        expected_emoji_hash="chosen",
    )
    assert value.success is False
    assert "Hook" in value.message


@pytest.mark.asyncio
async def test_bounded_cold_index_reports_partial_and_resumes(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "_space", lambda value: value.model_name)
    monkeypatch.setattr(retrieval, "MAX_INDEX_ITEMS_PER_QUERY", 1)
    client = SimpleNamespace(embed_text=AsyncMock(return_value=result([1, 0])))
    items = [emoji(tmp_path, name, name) for name in "abc"]
    index = retrieval.EmojiDescriptionIndex(tmp_path / "i")
    for expected in (1, 2, 3):
        value = await retrieval.query_registered_emojis("q", items, session_id="s", client=client, index=index)
        assert value.indexed == expected
        assert value.total == 3
        assert value.complete == (expected == 3)
    assert client.embed_text.await_count == 6


@pytest.mark.asyncio
async def test_concurrent_queries_do_not_duplicate_description_embeddings(monkeypatch, tmp_path):
    import asyncio
    from collections import Counter

    monkeypatch.setattr(retrieval, "_space", lambda value: value.model_name)
    calls = Counter()

    async def embed(text, **kwargs):
        calls[text] += 1
        await asyncio.sleep(0.02)
        return result([1, 0])

    items = [emoji(tmp_path, name, name) for name in "abc"]
    index = retrieval.EmojiDescriptionIndex(tmp_path / "i")
    client = SimpleNamespace(embed_text=embed)
    await asyncio.gather(
        *[retrieval.query_registered_emojis("q", items, session_id="s", client=client, index=index) for _ in range(2)]
    )
    assert calls["q"] == 2
    assert [calls[name] for name in "abc"] == [1, 1, 1]


@pytest.mark.asyncio
async def test_cancellation_releases_lease_and_keeps_completed_vectors(monkeypatch, tmp_path):
    import asyncio

    monkeypatch.setattr(retrieval, "_space", lambda value: value.model_name)
    started = asyncio.Event()

    async def embed(text, **kwargs):
        if text == "b":
            started.set()
            await asyncio.sleep(20)
        return result([1, 0])

    index = retrieval.EmojiDescriptionIndex(tmp_path / "i")
    items = [emoji(tmp_path, name, name) for name in "ab"]
    task = asyncio.create_task(
        retrieval.query_registered_emojis(
            "q",
            items,
            session_id="s",
            client=SimpleNamespace(embed_text=embed),
            index=index,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client = SimpleNamespace(embed_text=AsyncMock(return_value=result([1, 0])))
    value = await retrieval.query_registered_emojis("q", items, session_id="s", client=client, index=index)
    assert value.complete
    assert [call.args[0] for call in client.embed_text.await_args_list] == ["q", "b"]


@pytest.mark.asyncio
async def test_query_failure_never_exposes_provider_error_body(monkeypatch):
    from src.core.tooling import ToolInvocation
    from src.maisaka.builtin_tool import query_emoji

    monkeypatch.setattr(
        query_emoji, "query_registered_emojis", AsyncMock(side_effect=RuntimeError("secret-token-body"))
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(session_id="s"), build_failure_result=lambda *args, **kwargs: str((args, kwargs))
    )
    value = await query_emoji.handle_tool(ctx, ToolInvocation(tool_name="query_emoji", arguments={"query": "q"}))
    assert "secret-token-body" not in value


@pytest.mark.asyncio
async def test_cold_timeout_is_partial_not_failure_and_can_resume(monkeypatch, tmp_path):
    import asyncio

    normal_budget = retrieval.INDEX_SECONDS_PER_QUERY
    monkeypatch.setattr(retrieval, "_space", lambda value: value.model_name)
    monkeypatch.setattr(retrieval, "INDEX_SECONDS_PER_QUERY", 0.01)

    async def embed(text, **kwargs):
        if text != "q":
            await asyncio.sleep(1)
        return result([1, 0])

    index = retrieval.EmojiDescriptionIndex(tmp_path / "i")
    items = [emoji(tmp_path, "a", "a")]
    value = await retrieval.query_registered_emojis(
        "q", items, session_id="s", client=SimpleNamespace(embed_text=embed), index=index
    )
    assert not value.complete and value.indexed == 0
    # The forced 10ms timeout applies to the first attempt. Resume under the
    # normal budget so the test measures lease recovery, not filesystem speed.
    monkeypatch.setattr(retrieval, "INDEX_SECONDS_PER_QUERY", normal_budget)
    client = SimpleNamespace(embed_text=AsyncMock(return_value=result([1, 0])))
    value = await retrieval.query_registered_emojis("q", items, session_id="s", client=client, index=index)
    assert value.complete


def test_crashed_index_lease_expires(monkeypatch, tmp_path):
    index = retrieval.EmojiDescriptionIndex(tmp_path / "i")
    monkeypatch.setattr(retrieval.time, "time", lambda: 1000)
    assert index.reserve("space", "text", "first")
    assert not index.reserve("space", "text", "second")
    monkeypatch.setattr(retrieval.time, "time", lambda: 1000 + retrieval.INDEX_LEASE_SECONDS + 1)
    assert index.reserve("space", "text", "second")
    index.release("space", "text", "first")  # old owner cannot delete the new lease
    assert not index.reserve("space", "text", "third")


@pytest.mark.asyncio
async def test_ranking_runs_off_event_loop(monkeypatch, tmp_path):
    import threading

    main_thread = threading.get_ident()
    rank = retrieval._rank

    def checked_rank(*args):
        assert threading.get_ident() != main_thread
        return rank(*args)

    monkeypatch.setattr(retrieval, "_rank", checked_rank)
    monkeypatch.setattr(retrieval, "_space", lambda value: value.model_name)
    client = SimpleNamespace(embed_text=AsyncMock(return_value=result([1, 0])))
    hits = await retrieval.query_registered_emojis(
        "q",
        [emoji(tmp_path, "b", "same"), emoji(tmp_path, "a", "same")],
        session_id="s",
        client=client,
        index=retrieval.EmojiDescriptionIndex(tmp_path / "vectors.sqlite"),
    )
    assert [item.file_hash for item, _ in hits.hits] == ["a", "b"]


@pytest.mark.asyncio
async def test_cancel_during_reserve_releases_after_transaction(tmp_path):
    import asyncio
    import threading

    index = retrieval.EmojiDescriptionIndex(tmp_path / "vectors.sqlite")
    entered, proceed = threading.Event(), threading.Event()
    reserve = index.reserve

    def delayed_reserve(*args):
        entered.set()
        assert proceed.wait(5)
        return reserve(*args)

    index.reserve = delayed_reserve
    task = asyncio.create_task(retrieval._reserve(index, "space", "text", "owner"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(reserve, "space", "text", "next-owner")


@pytest.mark.asyncio
async def test_explicit_warmup_resumes_entire_library_without_chat_queries(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "_space", lambda value: value.model_name)
    client = SimpleNamespace(embed_text=AsyncMock(return_value=result([1, 0])))
    items = [emoji(tmp_path, str(i), f"description {i}") for i in range(12)]
    index = retrieval.EmojiDescriptionIndex(tmp_path / "vectors.sqlite")
    first = await retrieval.warm_registered_emojis(
        items, session_id="existing-session", client=client, index=index, max_items=5
    )
    assert (first.indexed, first.total, first.complete, first.hits) == (5, 12, False, [])
    second = await retrieval.warm_registered_emojis(
        items, session_id="existing-session", client=client, index=index, max_items=20
    )
    assert (second.indexed, second.total, second.complete) == (12, 12, True)
    assert client.embed_text.await_count == 14  # 12 descriptions plus 2 route probes
