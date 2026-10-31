"""Offline catalog/provenance regressions: real evidence, scope, edits and receipts."""

from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace as NS

import json

import pytest
from sqlmodel import Session, create_engine

from src.common.database.database_model import Expression, Jargon, Messages, ModifiedBy
from src.learners.expression_provenance import ExpressionProvenance, expression_identity, text_digest
from src.services import group_expression_catalog as catalog


@pytest.fixture
def storage(tmp_path, monkeypatch):
    from src.chat.replyer import maisaka_expression_selector as selector

    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}")
    for model in (Expression, Jargon, Messages):
        model.__table__.create(engine)

    @contextmanager
    def session(**kwargs):
        with Session(engine) as db:
            yield db
            if kwargs.get("auto_commit", True):
                db.commit()

    monkeypatch.setattr(catalog, "get_db_session", session)
    monkeypatch.setattr(selector, "get_db_session", session)
    monkeypatch.setattr(catalog, "ROOT_PATH", tmp_path)
    monkeypatch.setattr(catalog.ExpressionConfigUtils, "get_expression_config_for_chat", lambda sid: (True, True))
    monkeypatch.setattr(catalog.JargonConfigUtils, "get_jargon_config_for_chat", lambda sid: (True, True))
    monkeypatch.setattr(catalog.JargonConfigUtils, "resolve_jargon_group_scope", lambda sid: ({sid, "shared"}, False))
    monkeypatch.setattr(
        selector.maisaka_expression_selector, "_resolve_expression_group_scope", lambda sid: ({sid, "shared"}, False)
    )
    monkeypatch.setattr(selector.global_config.expression, "expression_checked_only", False)
    monkeypatch.setattr(catalog, "get_review_state", lambda identity: {"rejected": identity == 9})
    monkeypatch.setattr(catalog, "is_bot_self", lambda platform, uid: uid == "bot")
    monkeypatch.setattr(
        catalog.SessionMessage, "from_db_instance", lambda row: NS(raw_message=row.processed_plain_text)
    )
    # Preserve actual production extraction in a separate test; here raw_content is a test sentinel.
    monkeypatch.setattr(catalog, "extract_learning_text", lambda raw: raw or "")
    return engine, ExpressionProvenance(tmp_path / "data" / "joint_learning.sqlite3")


def expression(identity=1, **kwargs):
    return Expression(id=identity, situation="无奈认同", style="啊对对", content_list="[]", session_id="chat", **kwargs)


def message(identity, text, *, chat="chat", sender="u", **kwargs):
    return Messages(
        message_id=identity,
        session_id=chat,
        platform="qq",
        user_id=sender,
        user_nickname=sender,
        timestamp=datetime(2026, 1, 1),
        raw_content=b"test",
        processed_plain_text=text,
        **kwargs,
    )


def jargon(identity=1, *, chat="chat", refs=None, **kwargs):
    return Jargon(
        id=identity,
        content="群梗",
        meaning="本群用于打趣的说法",
        is_jargon=True,
        session_id_dict=json.dumps({chat: 1}),
        evidence_messages=json.dumps(refs or []),
        **kwargs,
    )


def ref(mid, **kwargs):
    return {"message_id": mid, "platform": "qq", **kwargs}


def save_source(store, row, mid="source", text="啊对对", contexts=None):
    store.save(
        row.id,
        expression_identity(row.situation, row.style, row.create_time.isoformat()),
        "chat",
        [ref(mid, text_hash=text_digest(text), context_refs=contexts or [])],
    )


@pytest.mark.asyncio
async def test_real_source_context_and_no_fabricated_legacy(storage):
    engine, store = storage
    with Session(engine) as db:
        row = expression()
        db.add_all([row, expression(2), message("source", "啊对对"), message("before", "你就说我赢没赢")])
        db.commit()
        save_source(store, row, contexts=[ref("before", text_hash=text_digest("你就说我赢没赢"))])
    records = await catalog.load_group_expression_catalog("chat")
    first, legacy = records
    assert first["id"] == "expression:1" and first["meaning"] == ""
    assert first["source_ids"] == ["source"]
    assert [(e["text"], e["is_source"]) for e in first["examples"]] == [("啊对对", True), ("你就说我赢没赢", False)]
    assert legacy["examples"] == legacy["source_ids"] == []
    assert records == await catalog.load_group_expression_catalog("chat")


@pytest.mark.asyncio
async def test_deleted_edited_and_reused_records_never_keep_old_evidence(storage):
    engine, store = storage
    with Session(engine) as db:
        row = expression()
        db.add_all([row, message("source", "啊对对")])
        db.commit()
        save_source(store, row)
    original = (await catalog.load_group_expression_catalog("chat"))[0]
    with Session(engine) as db:
        row = db.get(Expression, 1)
        row.style = "行吧"
        db.add(row)
        db.commit()
    changed = (await catalog.load_group_expression_catalog("chat"))[0]
    assert changed["phrase"] == "行吧" and changed["examples"] == []
    assert changed["version"] != original["version"]
    with Session(engine) as db:
        db.delete(db.get(Expression, 1))
        db.commit()
        row = expression()
        db.add(row)
        db.commit()
    assert (await catalog.load_group_expression_catalog("chat"))[0]["examples"] == []
    with Session(engine) as db:
        row = db.get(Expression, 1)
        save_source(store, row)
        source = db.get(Messages, 1)
        source.processed_plain_text = "编辑后不是证据"
        db.add(source)
        db.commit()
    assert (await catalog.load_group_expression_catalog("chat"))[0]["examples"] == []
    with Session(engine) as db:
        db.delete(db.get(Messages, 1))
        db.commit()
    assert (await catalog.load_group_expression_catalog("chat"))[0]["source_ids"] == []


@pytest.mark.asyncio
async def test_sharing_terms_never_shares_other_chat_raw_examples(storage):
    engine, store = storage
    refs = [[ref("shared"), ref("private"), ref("local"), ref("bot"), ref("notification"), ref("missing")]]
    with Session(engine) as db:
        db.add_all(
            [
                jargon(refs=refs),
                message("shared", "群梗 secret", chat="shared"),
                message("private", "群梗 private", chat="private"),
                message("local", "这个群梗真好笑"),
                message("bot", "群梗", sender="bot"),
                message("notification", "群梗", is_notify=True),
                jargon(2, chat="shared", refs=[[ref("shared")]]),
                jargon(3, chat="private"),
                jargon(4, chat="private", is_global=True),
                jargon(5, chat="chat", is_complete=True),
            ]
        )
        db.commit()
    records = await catalog.load_group_expression_catalog("chat")
    assert [r["id"] for r in records] == ["jargon:1", "jargon:2", "jargon:4", "jargon:5"]
    assert records[0]["source_ids"] == ["local"]
    assert [e["text"] for e in records[0]["examples"]] == ["这个群梗真好笑"]
    assert records[1]["examples"] == records[2]["examples"] == []


@pytest.mark.asyncio
async def test_expression_scope_curated_rejected_and_use_switches(storage, monkeypatch):
    from src.config.config import global_config

    engine, _ = storage
    with Session(engine) as db:
        db.add_all([expression(1, checked=True, modified_by=ModifiedBy.USER), expression(2), expression(9)])
        db.add(jargon())
        db.commit()
    assert [r["id"] for r in await catalog.load_group_expression_catalog("chat")] == [
        "expression:1",
        "expression:2",
        "jargon:1",
    ]
    monkeypatch.setattr(global_config.expression, "expression_checked_only", True)
    assert [r["id"] for r in await catalog.load_group_expression_catalog("chat")] == ["expression:1", "jargon:1"]
    monkeypatch.setattr(catalog.ExpressionConfigUtils, "get_expression_config_for_chat", lambda sid: (False, True))
    assert [r["id"] for r in await catalog.load_group_expression_catalog("chat")] == ["jargon:1"]
    monkeypatch.setattr(catalog.JargonConfigUtils, "get_jargon_config_for_chat", lambda sid: (False, True))
    assert await catalog.load_group_expression_catalog("chat") == []


def test_provenance_restart_and_identity_isolation(tmp_path):
    path = tmp_path / "joint.sqlite"
    store = ExpressionProvenance(path)
    assert store.load("chat") == {}
    store.save(1, "version", "chat", [ref("source")])
    assert ExpressionProvenance(path).load("chat")[1]["sources"] == [ref("source")]
    assert store.load("other") == {}


@pytest.mark.asyncio
async def test_unambiguous_matching_jargon_enriches_without_losing_identity(storage):
    engine, _ = storage
    with Session(engine) as db:
        term = jargon(refs=[[ref("local")]])
        term.content = "啊对对"
        db.add_all([expression(), term, message("local", "啊对对，你说得都对")])
        db.commit()
    records = await catalog.load_group_expression_catalog("chat")
    assert len(records) == 2
    fused = records[0]
    assert fused["id"] == "expression:1"
    assert fused["related_ids"] == ["jargon:1"]
    assert fused["meaning"] == "本群用于打趣的说法"
    assert (
        fused["meaning_source"] == "linked_jargon_not_situation_verified"
    )  # linked spelling, not confirmed sense equivalence
    assert fused["related_versions"] == {"jargon:1": records[1]["version"]}
    assert fused["situation"] == "无奈认同"
    assert fused["source_ids"] == []
    assert fused["examples"] == []  # spelling association cannot establish usage in this situation
    with Session(engine) as db:
        conflicting = jargon(2)
        conflicting.content, conflicting.meaning = "啊对对", "真诚表示肯定"
        db.add(conflicting)
        db.commit()
    records = await catalog.load_group_expression_catalog("chat")
    assert len(records) == 3 and records[0]["meaning"] == ""
    assert "related_ids" not in records[0]
    assert records[0]["version"] != fused["version"]


def test_catalog_filters_quotes_from_own_text():
    from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
    from src.learners.learning_text import extract_learning_text

    # Quote/media components are not top-level TextComponent instances.
    seq = NS(components=[TextComponent(text="own words"), MessageSequence(components=[TextComponent(text="quote")])])
    assert extract_learning_text(seq) == "own words"


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True])
async def test_actual_write_receipt_persists_before_index_failure(storage, tmp_path, monkeypatch, abort):
    from unittest.mock import AsyncMock
    from src.common.data_models.expression_data_model import MaiExpression
    from src.learners import expression_learner as learner_module
    from src.learners import joint_learning as joint

    engine, store = storage
    with Session(engine) as db:
        db.add(message("source", "啊对对"))
        db.commit()
    learner = learner_module.ExpressionLearner("chat")
    monkeypatch.setattr(learner_module.prompt_manager, "get_prompt", lambda name: NS(add_context=lambda *a: None))
    monkeypatch.setattr(learner_module.prompt_manager, "render_prompt", AsyncMock(return_value="unused"))
    monkeypatch.setattr(learner, "_prepare_learning_messages", lambda messages: messages)
    monkeypatch.setattr(learner, "_filter_expressions", lambda candidates, messages: [(a, b) for a, b, c in candidates])
    monkeypatch.setattr(learner, "_find_similar_expression", lambda *args, **kwargs: None)
    monkeypatch.setattr(learner, "_get_session_display_name", lambda sid: sid)
    monkeypatch.setattr(
        learner, "_get_runtime_manager", lambda: NS(invoke_hook=AsyncMock(return_value=NS(aborted=abort, kwargs={})))
    )
    monkeypatch.setattr(
        learner, "_sync_expression_vector_index_batch", AsyncMock(side_effect=RuntimeError("index down"))
    )
    monkeypatch.setattr(joint.global_config.expression, "expression_self_reflect", False)
    monkeypatch.setattr(joint, "extract_learning_text", lambda raw: raw)

    async def upsert(situation, style, **kwargs):
        with Session(engine) as db:
            row = expression()
            db.add(row)
            db.commit()
            return MaiExpression.from_db_instance(row)

    monkeypatch.setattr(learner, "_upsert_expression_to_db", upsert)
    service = joint.JointLearningCoordinator(
        NS(
            session_id="chat",
            _enable_expression_learning=True,
            _expression_learner=learner,
        ),
        store.path,
    )
    service.journal.initialize()
    service.journal.enqueue("chat", ["source"], 0)
    batch = service.journal.next_batch("chat", 1800)
    operation = service.process_expressions(
        batch,
        [NS(message_id="source", platform="qq", raw_message="啊对对")],
        [{"situation": "无奈认同", "style": "啊对对", "id": "1"}],
    )
    if abort:
        await operation
        assert store.load("chat") == {}
        assert await catalog.load_group_expression_catalog("chat") == []
    else:
        with pytest.raises(RuntimeError, match="index down"):
            await operation
        assert ExpressionProvenance(store.path).load("chat")[1]["sources"][0]["message_id"] == "source"
        assert ExpressionProvenance(store.path).load("chat")[1]["sources"][0]["example"] == "啊对对"
        records = await catalog.load_group_expression_catalog("chat")
        assert records[0]["examples"][0]["text"] == "啊对对"
        assert records[0]["source_ids"] == ["source"]


@pytest.mark.asyncio
async def test_snapshot_must_match_live_source_and_metadata_upgrade_is_not_new_evidence(storage, monkeypatch):
    from src.learners import jargon_miner as module

    engine, _ = storage
    original = ref("m")
    snapshot = ref("m", example="群梗真有意思", text_hash=text_digest("群梗真有意思"))
    with Session(engine) as db:
        term = jargon(refs=[[original]], count=4)
        term.session_id_dict = json.dumps({"chat": 4})
        db.add_all([term, message("m", "群梗真有意思")])
        db.commit()
        term = db.get(Jargon, 1)
        miner = module.JargonMiner.__new__(module.JargonMiner)
        miner.session_id = "chat"
        monkeypatch.setattr(module, "get_db_session", catalog.get_db_session)
        miner._update_jargon(term, [[snapshot]], strict_persistence=True)
    with Session(engine) as db:
        term = db.get(Jargon, 1)
        assert term.count == 4 and json.loads(term.session_id_dict) == {"chat": 4}
        assert json.loads(term.evidence_messages)[0][0]["example"] == "群梗真有意思"
    assert (await catalog.load_group_expression_catalog("chat"))[0]["examples"][0]["text"] == "群梗真有意思"
    with Session(engine) as db:
        term = db.get(Jargon, 1)
        bad = {**snapshot, "example": "invented"}
        term.evidence_messages = json.dumps([[bad]])
        db.add(term)
        db.commit()
    assert (await catalog.load_group_expression_catalog("chat"))[0]["examples"] == []
