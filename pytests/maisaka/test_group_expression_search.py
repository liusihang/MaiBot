from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.common.data_models.embedding_service_data_models import EmbeddingResult
from src.services import group_expression_search as search


def record(identifier, phrase, meaning="", situation=""):
    return {
        "id": identifier,
        "kind": identifier.split(":")[0],
        "phrase": phrase,
        "meaning": meaning,
        "situation": situation,
        "examples": [],
        "source_ids": [],
        "version": "v1",
    }


def embedding(vector, model="test"):
    return EmbeddingResult(embedding=vector, model_name=model, model_identifier=model, api_provider="test")


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(search.vector_cache, "_space", lambda value: value.model_name)
    # Filesystem/thread scheduling is not what these deterministic rank tests time.
    monkeypatch.setattr(search, "INDEX_SECONDS_PER_QUERY", 60)
    monkeypatch.setattr(search.vector_cache, "INDEX_SECONDS_PER_QUERY", 60)
    index = search.vector_cache.EmojiDescriptionIndex(tmp_path / "language.sqlite")

    def install(records, vectors=None):
        loader = AsyncMock(return_value=records)
        monkeypatch.setattr(search, "load_group_expression_catalog", loader)
        client = SimpleNamespace(
            embed_text=AsyncMock(side_effect=lambda text, **_: embedding(vectors[text] if vectors else [1, 0]))
        )
        return client, index, loader

    return install


def test_chinese_bigrams_and_bm25():
    assert search.tokenize("表示佩服 NB") == ["表示", "示佩", "佩服", "nb"]
    scores = search.bm25_scores(["表示佩服 nb", "表示失败 gg", "祝你生日快乐"], "佩服")
    assert scores[0] > 0 and scores[1:] == [0, 0]


def test_rrf_uses_independent_ranks_not_raw_scores():
    records = [record("expression:1", "a"), record("jargon:2", "b"), record("jargon:3", "c")]
    ranked = search._rank(
        records, ["a", "b", "c"], [1000000, 1, 0], {"a": [0.5, 0.866], "b": [1, 0], "c": [0.8, 0.6]}, [1, 0], 3
    )
    # b: lexical rank2 + semantic rank1 beats a: lexical rank1 + semantic rank3.
    assert ranked[0]["id"] == "jargon:2"
    assert ranked[0]["ranking"]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)


@pytest.mark.asyncio
async def test_full_hybrid_finds_intent_without_literal_overlap_and_preserves_examples(setup):
    records = [record("expression:1", "逆天", "离谱到让人惊讶"), record("jargon:2", "开香槟", "提前庆祝")]
    records[0]["examples"] = [{"message_id": "m1", "text": "这操作逆天", "sender_id": "u", "time": 1}]
    query = "unbelievable"
    docs = [search.semantic_document(item) for item in records]
    client, index, loader = setup(records, {query: [1, 0], docs[0]: [1, 0], docs[1]: [0, 1]})
    result = await search.search_group_expressions("chat-A", query, client=client, index=index)
    assert loader.await_count == 2
    loader.assert_awaited_with("chat-A")
    assert result["coverage"]["mode"] == "hybrid"
    assert result["coverage"]["complete"]
    assert result["candidates"][0]["id"] == "expression:1"
    assert result["candidates"][0]["ranking"]["bm25_rank"] is None
    assert result["candidates"][0]["examples"] == records[0]["examples"]


@pytest.mark.asyncio
async def test_cache_reuse_and_content_edit_and_model_change(setup):
    records = [record("expression:1", "nb", "佩服")]
    client, index, _ = setup(records)
    await search.search_group_expressions("s", "佩服", client=client, index=index)
    assert client.embed_text.await_count == 2
    await search.search_group_expressions("s", "佩服", client=client, index=index)
    assert client.embed_text.await_count == 3
    records[0]["meaning"] = "赞叹"
    await search.search_group_expressions("s", "佩服", client=client, index=index)
    assert client.embed_text.await_count == 5
    client.embed_text = AsyncMock(return_value=embedding([1, 0], "new"))
    result = await search.search_group_expressions("s", "佩服", client=client, index=index)
    assert client.embed_text.await_count == 2
    assert result["coverage"]["embedding_space"] == "new"


@pytest.mark.asyncio
async def test_cold_bound_then_warm_complete_and_no_registration_first_eight_bias(setup):
    records = [record(f"expression:{i}", f"saying{i}") for i in range(20)]
    records[-1]["meaning"] = "celebrate"
    client, index, _ = setup(records)
    result = await search.search_group_expressions("s", "celebrate", client=client, index=index)
    assert result["coverage"]["indexed"] == 8
    assert result["coverage"]["mode"] == "partial_hybrid"
    assert client.embed_text.await_count == 9
    assert any(
        item["id"] == "expression:19" and item["ranking"]["semantic_rank"] is not None for item in result["candidates"]
    )
    progress = await search.warm_group_expression_index("s", max_items=30, client=client, index=index)
    assert progress["complete"] and progress["indexed"] == 20
    before = client.embed_text.await_count
    result = await search.search_group_expressions("s", "celebrate", client=client, index=index)
    assert result["coverage"]["mode"] == "hybrid"
    assert client.embed_text.await_count == before + 1


@pytest.mark.asyncio
async def test_homonyms_kept_and_scope_loaded_afresh(setup):
    records = [record("expression:1", "绝了", "赞赏"), record("jargon:2", "绝了", "无奈")]
    client, index, loader = setup(records)
    result = await search.search_group_expressions("s", "绝了", client=client, index=index)
    assert len(result["candidates"]) == 2
    # A removed/disallowed catalog record must not be revived by vector cache.
    loader.return_value = records[:1]
    result = await search.search_group_expressions("s", "绝了", client=client, index=index)
    assert [item["id"] for item in result["candidates"]] == ["expression:1"]


@pytest.mark.asyncio
async def test_route_mismatch_is_explicit_lexical_only_never_mixed(setup):
    client, index, _ = setup([record("expression:1", "nb", "佩服")])
    client.embed_text = AsyncMock(side_effect=[embedding([1, 0]), embedding([1, 0], "different")])
    result = await search.search_group_expressions("s", "佩服", client=client, index=index)
    assert result["coverage"]["mode"] == "lexical_only"
    assert result["coverage"]["embedding_error"] == "ValueError"
    assert result["candidates"][0]["ranking"]["semantic_rank"] is None


@pytest.mark.asyncio
async def test_unavailable_embedding_and_empty_catalog_are_honest(setup):
    client, index, loader = setup([record("expression:1", "nb", "佩服")])
    client.embed_text = AsyncMock(side_effect=RuntimeError("private provider error"))
    result = await search.search_group_expressions("s", "佩服", client=client, index=index)
    assert result["coverage"]["mode"] == "lexical_only"
    assert result["coverage"]["embedding_error"] == "RuntimeError"
    assert "private" not in str(result)
    loader.return_value = []
    result = await search.search_group_expressions("s", "佩服", client=client, index=index)
    assert result["coverage"]["mode"] == "empty"
    assert result["candidates"] == []


@pytest.mark.asyncio
async def test_low_cosine_semantic_only_hit_survives_for_agent_applicability_check(setup):
    item = record("jargon:1", "提前庆祝")
    client, index, _ = setup([item], {"sympathy": [1, 0], search.semantic_document(item): [0.2, 0.9797958971132712]})
    result = await search.search_group_expressions("s", "sympathy", client=client, index=index)
    assert result["coverage"]["complete"]
    assert result["candidates"][0]["ranking"]["cosine_score"] == pytest.approx(0.2)
    assert result["candidates"][0]["ranking"]["bm25_rank"] is None
    assert result["candidates"][0]["ranking"]["semantic_rank"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent,context,limit", [("", "", 5), ("a" * 1001, "", 5), ("x", "y" * 2001, 5), ("x", "", 0), ("x", "", True)]
)
async def test_input_bounds(intent, context, limit):
    with pytest.raises(ValueError):
        await search.search_group_expressions("s", intent, context, limit)


def test_document_fields_bounded_and_real_examples_only():
    item = record("expression:1", "x" * 5000, "y" * 5000, "z" * 5000)
    item["examples"] = [{"text": "real source " + "e" * 5000} for _ in range(20)]
    text = search.semantic_document(item)
    assert len(text) <= search.MAX_DOCUMENT_CHARS
    assert "meaning: y" in text and "situation: z" in text and "real source" in text


def test_candidate_payload_bounded_without_inventing_evidence():
    item = record("expression:1", "x" * 5000, "y" * 5000, "z" * 5000)
    item["examples"] = [{"message_id": str(i), "text": "real " + "a" * 3000} for i in range(10)]
    item["source_ids"] = list(range(100))
    result = search._bounded_candidate(item)
    assert len(result["meaning"]) == 1000 and len(result["examples"]) == 3
    assert len(result["examples"][0]["text"]) == 600 and result["examples"][0]["truncated"]
    assert len(result["source_ids"]) == 20
    assert result["examples"][0]["text"] == item["examples"][0]["text"][:600]
    assert "meaning" in result["truncated_fields"]


def test_semantic_structural_labels_are_not_lexical_terms():
    _, scores, _ = search._prepare([record("expression:1", "nb", "佩服")], "meaning examples phrase situation")
    assert scores == [0]


@pytest.mark.asyncio
async def test_cancelled_search_releases_lease_and_next_search_resumes(setup):
    import asyncio
    import sqlite3

    client, index, _ = setup([record("expression:1", "nb", "佩服")])
    started = asyncio.Event()

    async def blocking_document(text, **_):
        if text == "佩服":
            return embedding([1, 0])
        started.set()
        await asyncio.Event().wait()

    client.embed_text = AsyncMock(side_effect=blocking_document)
    task = asyncio.create_task(search.search_group_expressions("s", "佩服", client=client, index=index))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with sqlite3.connect(index.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0
    client.embed_text = AsyncMock(return_value=embedding([1, 0]))
    result = await search.search_group_expressions("s", "佩服", client=client, index=index)
    assert result["coverage"]["mode"] == "hybrid"
    assert client.embed_text.await_count == 2


@pytest.mark.asyncio
async def test_total_serialized_result_budget_disclosed(setup):
    import json

    records = [record(f"expression:{i}", "x" * 400, "meaning" * 200, "situation" * 200) for i in range(5)]
    for item in records:
        item["examples"] = [{"text": "e" * 600} for _ in range(3)]
    client, index, _ = setup(records)
    result = await search.search_group_expressions("s", "x", client=client, index=index)
    assert len(json.dumps(result["candidates"], ensure_ascii=False)) <= search.MAX_RESULT_CHARS
    assert result["coverage"]["results_truncated"]
    assert 0 < len(result["candidates"]) < 5


@pytest.mark.asyncio
async def test_service_result_limit_matches_tool_limit():
    with pytest.raises(ValueError):
        await search.search_group_expressions("s", "hello", limit=7)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["delete", "edit"])
async def test_catalog_revalidated_after_embedding_drops_removed_or_changed_evidence(setup, change):
    original = record("expression:1", "秘密梗", "原私密解释")
    replacement = {**original, "meaning": "修改后的解释", "version": "v2"}
    client, index, loader = setup([original])
    loader.side_effect = [[original], [] if change == "delete" else [replacement]]
    result = await search.search_group_expressions("s", "秘密", client=client, index=index)
    assert result["candidates"] == []
    assert result["coverage"]["catalog_changed"]
    assert result["coverage"]["stale_records_removed"] == 1
    if change == "edit":
        assert not result["coverage"]["complete"] and not result["coverage"]["lexical_complete"]


@pytest.mark.asyncio
async def test_default_limit_returns_six_combined_expression_and_jargon_records(setup):
    records = [record(("expression:" if i % 2 else "jargon:") + str(i), "庆祝" + str(i)) for i in range(8)]
    client, index, _ = setup(records)
    result = await search.search_group_expressions("s", "庆祝", client=client, index=index)
    assert len(result["candidates"]) == 6
    assert {item["kind"] for item in result["candidates"]} == {"expression", "jargon"}
