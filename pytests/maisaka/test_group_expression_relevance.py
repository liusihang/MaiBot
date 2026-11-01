"""A top-k rank is not evidence that language material fits the requested intent."""

import pytest

from src.services import group_expression_search as search


def item(identifier, phrase):
    return {
        "id": identifier,
        "kind": identifier.split(":")[0],
        "phrase": phrase,
        "meaning": "",
        "situation": "",
        "examples": [],
        "source_ids": [],
        "version": "v1",
    }


def test_weak_semantics_cannot_be_rescued_by_large_keyword_score():
    records = [item("expression:1", "懂得自然懂"), item("jargon:2", "吊打")]
    result = search._rank(
        records,
        ["a", "b"],
        [10000, 100],
        {"a": [0.4, 0.9165], "b": [0.5, 0.866]},
        [1, 0],
        6,
        query="严肃介绍数据库索引",
    )
    assert result == []


def test_relevance_filter_returns_fewer_than_limit():
    records = [item("jargon:1", "轮椅"), item("jargon:2", "刮痧"), item("jargon:3", "脑测")]
    result = search._rank(
        records,
        ["a", "b", "c"],
        [10, 9, 8],
        {"a": [0.8, 0.6], "b": [0.4, 0.9165], "c": [0.3, 0.9539]},
        [1, 0],
        6,
        query="低门槛强势打法",
    )
    assert [row["id"] for row in result] == ["jargon:1"]


def test_missing_vectors_require_actual_phrase_not_meaning_overlap():
    records = [item("jargon:1", "轮椅"), item("jargon:2", "脑测")]
    result = search._rank(records, ["a", "b"], [1, 10000], {}, None, 6, query="用轮椅形容轻松过关")
    assert [row["id"] for row in result] == ["jargon:1"]
    assert search._rank(records, ["a", "b"], [1, 10000], {}, None, 6, query="客观介绍通关设置") == []


def test_quoted_expression_can_match_literally_without_vectors():
    records = [item("expression:1", "用“云玩家还锐评上了”调侃")]
    assert search._rank(records, ["a"], [1], {}, None, 6, query="云玩家还锐评上了")[0]["id"] == "expression:1"


def test_ascii_literal_match_has_word_boundaries():
    records = [item("jargon:1", "nb")]
    assert search._rank(records, ["a"], [1], {}, None, 6, query="sunbeam") == []
    assert search._rank(records, ["a"], [1], {}, None, 6, query="NB，厉害")


def test_literal_hit_cannot_override_a_measured_weak_semantic_match():
    records = [item("jargon:1", "轮椅")]
    assert (
        search._rank(
            records,
            ["a"],
            [10000],
            {"a": [0.3, 0.9539]},
            [1, 0],
            6,
            query="轮椅这个字出现在文件名中，请正式翻译整个文件",
        )
        == []
    )


def test_source_examples_do_not_change_search_documents_or_keyword_ranks():
    record = item("expression:1", "用倒装感叹表达惊讶")
    record["situation"] = "惊讶于事情突然发生"
    before = search._prepare([record], "叶师傅等破绽")
    record["examples"] = [{"text": "叶师傅等破绽，来了我超"}]
    after = search._prepare([record], "叶师傅等破绽")
    assert before == after


def test_topic_match_cannot_override_an_incompatible_communicative_situation():
    record = item("expression:1", "没交互，纯等他出破绽")
    record["situation"] = "吐槽战斗玩法单调"
    document = search.semantic_document(record)
    situation = search.situation_document(record)
    result = search._rank(
        [record], [document], [10000],
        {document: [0.95, 0.3122], situation: [0.4, 0.9165]},
        [1, 0], 6, query="俏皮表示自己会主动互动，接等破绽的梗",
    )
    assert result == []


def test_expression_with_missing_situation_vector_is_not_semantically_ready():
    record = item("expression:1", "用反话调侃装懂")
    record["situation"] = "反讽装懂"
    document = search.semantic_document(record)
    vectors = {document: [1, 0]}
    assert search._rank([record], [document], [100], vectors, [1, 0], 6) == []
    assert search._indexed_count([record], [document], vectors) == 0


@pytest.mark.asyncio
async def test_style_intent_is_separate_from_topic_context(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from src.common.data_models.embedding_service_data_models import EmbeddingResult

    records = [item("expression:1", "接招的说法"), item("jargon:2", "轮椅")]
    records[0]["situation"] = "接受挑战"
    monkeypatch.setattr(search, "load_group_expression_catalog", AsyncMock(return_value=records))
    monkeypatch.setattr(search.vector_cache, "_space", lambda result: result.model_name)
    client = SimpleNamespace(embed_text=AsyncMock(return_value=EmbeddingResult(
        embedding=[1, 0], model_name="test", model_identifier="test", api_provider="test"
    )))
    index = search.vector_cache.EmojiDescriptionIndex(tmp_path / "vectors.sqlite")
    result = await search.search_group_expressions("chat", "接受挑战", "游戏里在等破绽", client=client, index=index)
    inputs = [call.args[0] for call in client.embed_text.await_args_list]
    assert "接受挑战" in inputs
    assert "接受挑战\n游戏里在等破绽" in inputs
    assert result["coverage"]["complete"]
    assert result["candidates"][0]["ranking"]["situation_score"] == pytest.approx(1)


@pytest.mark.asyncio
async def test_maintenance_skips_unchanged_and_rewarms_changed_catalog(monkeypatch):
    from unittest.mock import AsyncMock

    records = [item("jargon:1", "轮椅")]
    load = AsyncMock(return_value=records)
    warm = AsyncMock(return_value={"complete": True})
    monkeypatch.setattr(search, "load_group_expression_catalog", load)
    monkeypatch.setattr(search, "warm_group_expression_index", warm)
    maintainer = search.GroupLanguageIndexMaintainer("chat")
    await maintainer.sync_once()
    await maintainer.sync_once()
    assert warm.await_count == 1
    records[0]["version"] = "edited"
    await maintainer.sync_once()
    assert warm.await_count == 2
    load.return_value = []
    await maintainer.sync_once()
    await maintainer.sync_once()
    assert warm.await_count == 2


@pytest.mark.asyncio
async def test_maintenance_retries_partial_warmup_and_edits_during_warmup(monkeypatch):
    from unittest.mock import AsyncMock

    records = [item("jargon:1", "轮椅")]
    load = AsyncMock(return_value=records)
    warm = AsyncMock(side_effect=[{"complete": False}, {"complete": True}, {"complete": True}])
    monkeypatch.setattr(search, "load_group_expression_catalog", load)
    monkeypatch.setattr(search, "warm_group_expression_index", warm)
    maintainer = search.GroupLanguageIndexMaintainer("chat")
    await maintainer.sync_once()
    assert maintainer.version is None
    load.side_effect = [records, [{**records[0], "version": "new"}]]
    await maintainer.sync_once()
    assert maintainer.version is None
    load.side_effect = None
    await maintainer.sync_once()
    assert maintainer.version == maintainer.signature(records)


@pytest.mark.asyncio
async def test_maintenance_start_is_idempotent_and_stop_cancels(monkeypatch):
    import asyncio

    maintainer = search.GroupLanguageIndexMaintainer("chat")
    started = asyncio.Event()

    async def run():
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(maintainer, "run", run)
    maintainer.start()
    task = maintainer.task
    maintainer.start()
    assert maintainer.task is task
    await started.wait()
    await maintainer.stop()
    assert task.cancelled() and maintainer.task is None


@pytest.mark.asyncio
async def test_service_maintains_quiet_sessions_and_cleans_up_workers(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    started = asyncio.Event()
    worker = SimpleNamespace(start=lambda: started.set(), stop=AsyncMock())
    monkeypatch.setattr(search, "_language_session_ids", lambda: ["stored-quiet-chat"])
    constructed = []

    def create(session_id):
        constructed.append(session_id)
        return worker

    monkeypatch.setattr(search, "GroupLanguageIndexMaintainer", create)
    task = asyncio.create_task(search.maintain_known_group_language_indexes())
    await asyncio.wait_for(started.wait(), 2)
    assert constructed == ["stored-quiet-chat"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    worker.stop.assert_awaited_once()


def test_session_discovery_uses_existing_session_ids_only(monkeypatch):
    from contextlib import contextmanager
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine
    from src.common.database import database
    from src.common.database.database_model import ChatSession, Expression, Jargon

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all([ChatSession(session_id=sid, platform="qq") for sid in ["quiet", "jargon-chat", "unrelated"]])
        db.add_all(
            [
                Expression(situation="场景", style="说法", content_list="[]", session_id=sid)
                for sid in ["quiet", "orphan"]
            ]
        )
        db.add(Jargon(content="词", meaning="义", session_id_dict='{"jargon-chat":1,"orphan":1}'))
        db.commit()

    @contextmanager
    def session(**kwargs):
        with Session(engine) as db:
            yield db

    monkeypatch.setattr(database, "get_db_session", session)
    assert search._language_session_ids() == ["jargon-chat", "quiet"]
    engine.dispose()
