"""A top-k rank is not evidence that language material fits the requested intent."""

import pytest
import math

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


@pytest.mark.parametrize("phrase,situation,score,usage_score", [
    ("用工程步骤描述离谱操作", "调侃把人物当作资源", 0.696, 0.732),
    ("用剥夺讨论权开玩笑", "调侃缺乏挑战精神", 0.674, 0.709),
    ("承认刚才随口乱说", "自嘲收回先前的判断", 0.719, 0.783),
])
def test_deployed_embedding_near_misses_can_return_empty(phrase, situation, score, usage_score):
    """Observed Qwen scores for generic banter and a mismatched speech act."""
    record = item("expression:1", phrase)
    record["situation"] = situation
    document = search.semantic_document(record)
    vectors = {document: [score, math.sqrt(1 - score * score)],
               situation: [usage_score, math.sqrt(1 - usage_score * usage_score)]}
    assert search._rank([record], [document], [100], vectors, [1, 0], 6, query="随口答应 万能回复") == []


@pytest.mark.parametrize("score,usage_score", [(0.789, 0.905), (0.766, 0.773), (0.732, 0.935), (0.752, 0.994)])
def test_deployed_clear_expression_matches_survive_calibration(score, usage_score):
    record = item("expression:1", "可借鉴的说法")
    record["situation"] = "与表达目的相符的场景"
    document = search.semantic_document(record)
    vectors = {document: [score, math.sqrt(1 - score * score)],
               record["situation"]: [usage_score, math.sqrt(1 - usage_score * usage_score)]}
    assert search._rank([record], [document], [0], vectors, [1, 0], 6)


def test_specific_usage_words_preserve_a_moderate_semantic_match():
    record = item("expression:1", "用云玩家还锐评上了调侃")
    record["situation"] = "吐槽纸上谈兵时"
    document = search.semantic_document(record)
    vectors = {document: [0.698, math.sqrt(1 - 0.698 ** 2)],
               record["situation"]: [0.767, math.sqrt(1 - 0.767 ** 2)]}
    assert search._rank([record], [document], [10], vectors, [1, 0], 6, query="没玩过 装懂 纸上谈兵")
    assert search._rank([record], [document], [10], vectors, [1, 0], 6, query="顺着群友的话调侃一句") == []


def test_common_banter_words_are_not_specific_usage_support():
    records = [item(f"expression:{i}", "一句说法") for i in range(12)]
    for record in records:
        record["situation"] = "调侃对方"
    assert search._usage_keyword_support(records, "调侃") == set()
    records[0]["situation"] = "调侃纸上谈兵"
    assert search._usage_keyword_support(records, "纸上谈兵") == {0}


def test_jargon_semantic_near_miss_without_meaning_support_is_empty():
    record = item("jargon:1", "康康")
    record["meaning"] = "看看的谐音，用来轻松表示准备查看"
    document = search.semantic_document(record)
    assert search._rank([record], [document], [0], {document: [0.630, math.sqrt(1 - 0.630 ** 2)], record["meaning"]: [0.4, math.sqrt(1 - 0.4 ** 2)]},
                        [1, 0], 6, query="万能回复") == []


def test_jargon_specific_meaning_words_preserve_a_borderline_match():
    record = item("jargon:1", "靶场战神")
    record["meaning"] = "调侃靶场测试很好，实战却不一定好用的武器或配置"
    document = search.semantic_document(record)
    assert search._rank([record], [document], [10], {document: [0.631, math.sqrt(1 - 0.631 ** 2)], record["meaning"]: [1, 0]},
                        [1, 0], 6, query="测试强 实战差")


def test_jargon_missing_meaning_vector_is_not_semantically_ready():
    record = item("jargon:1", "刮痧")
    record["meaning"] = "攻击伤害低"
    document = search.semantic_document(record)
    assert search._rank([record], [document], [100], {document: [1, 0]}, [1, 0], 6, query="伤害低") == []
    assert search._indexed_count([record], [document], {document: [1, 0]}) == 0


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


def test_topic_context_cannot_create_a_jargon_hit_without_intent_relevance():
    record = item("jargon:1", "靶场战神")
    assert search._rank(
        [record], ["document"], [100], {"document": [0, 1]}, [1, 0], 6,
        query="澄清假设题没有指具体的人", jargon_query_vector=[0, 1], context="游戏测试题",
    ) == []


def test_topic_context_can_rank_jargon_that_already_matches_intent():
    records = [item("jargon:1", "甲"), item("jargon:2", "乙")]
    ranked = search._rank(
        records, ["a", "b"], [0, 0],
        {"a": [0.75, 0.6614378278], "b": [0.65, 0.7599342077]},
        [1, 0], 6, query="核心含义", jargon_query_vector=[0, 1], context="话题领域",
    )
    assert [row["id"] for row in ranked] == ["jargon:2", "jargon:1"]


def test_missing_jargon_vector_requires_literal_intent_not_only_background():
    record = item("jargon:1", "靶场战神")
    assert search._rank(
        [record], ["document"], [100], {}, None, 6,
        query="澄清假设题", context="之前有人说过靶场战神",
    ) == []
    assert search._rank(
        [record], ["document"], [100], {}, None, 6,
        query="靶场战神", context="游戏",
    )[0]["id"] == "jargon:1"


@pytest.mark.asyncio
async def test_search_checks_jargon_against_intent_before_context(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from src.common.data_models.embedding_service_data_models import EmbeddingResult

    record = item("jargon:1", "靶场战神")
    query, context = "澄清假设题没有指具体的人", "游戏测试题"
    document = search.semantic_document(record)
    values = {query: [1, 0], query + "\n" + context: [0, 1], document: [0, 1]}
    monkeypatch.setattr(search, "load_group_expression_catalog", AsyncMock(return_value=[record]))
    monkeypatch.setattr(search.vector_cache, "_space", lambda result: result.model_name)
    client = SimpleNamespace(embed_text=AsyncMock(side_effect=lambda text, **_: EmbeddingResult(
        embedding=values[text], model_name="test", model_identifier="test", api_provider="test",
    )))
    index = search.vector_cache.EmojiDescriptionIndex(tmp_path / "vectors.sqlite")
    result = await search.search_group_expressions("chat", query, context, client=client, index=index)
    assert result["coverage"]["complete"]
    assert result["candidates"] == []


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
    situation = search.usage_document(record)
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
