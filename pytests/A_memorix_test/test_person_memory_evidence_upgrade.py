from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from src.common.data_models.message_component_data_model import TextComponent
from src.services.memory_records import direct_self_evidence, fact_classification, source_evidence


def test_reply_context_never_becomes_own_evidence():
    reply = NS(
        format_name="reply",
        target_message_id="previous",
        target_message_sender_id="other",
        target_message_content="我在北京工作",
    )
    source = NS(
        message_id="new",
        timestamp=datetime(2026, 10, 31, tzinfo=timezone.utc),
        raw_message=NS(components=[reply, TextComponent(text="我这周在上海出差")]),
    )
    record = source_evidence(source)
    assert record["time"] == "2026-10-31T00:00:00+00:00"
    assert record["reply_to"][0]["sender_id"] == "other"
    assert record["text"] == "我这周在上海出差"
    assert direct_self_evidence({"evidence_message_id": "new", "evidence_quote": "我在北京工作"}, [source]) is None


def test_temporal_and_uncertain_never_become_stable_profile_facts():
    assert fact_classification({"stability": "temporal", "profile_section": "identity_settings"}) == {
        "authority": "direct_user",
        "stability": "temporal",
        "profile_section": "recent_interactions",
    }
    assert (
        fact_classification({"stability": "uncertain", "profile_section": "stable_facts"})["profile_section"]
        == "uncertain_notes"
    )
    assert fact_classification({"stability": "invented"}) is None
    assert fact_classification({"stability": "stable", "profile_section": "invented"}) is None


def test_old_extraction_receipt_remains_readable():
    assert fact_classification({"profile_section": "interaction_preferences"})["stability"] == "stable"


def _claim(store, text, message_id, chat_id="chat", authority="direct_user"):
    paragraph = store.add_paragraph(
        content=text,
        source="person_fact:alice",
        knowledge_type="factual",
        metadata={
            "source_type": "person_fact",
            "chat_id": chat_id,
            "dedup_managed": True,
            "dedup_fact_texts": [text],
            "evidence_message_ids": [message_id],
        },
    )
    claim = store.upsert_fact_claim(
        scope_type="person",
        scope_id="alice",
        fact_key="statement:" + message_id,
        value_text=text,
        authority=authority,
        evidence_type="paragraph",
        evidence_id=paragraph,
    )
    return claim, paragraph


def test_explicit_correction_is_scoped_versioned_and_idempotent(tmp_path):
    from src.A_memorix.core.storage.metadata_store import MetadataStore

    store = MetadataStore(data_dir=tmp_path)
    store.connect()
    try:
        old, old_paragraph = _claim(store, "Alice 喜欢喝茶", "old")
        new, _ = _claim(store, "Alice 不再喝茶", "new")
        kwargs = dict(
            previous=old, new_claim_id=new["claim_id"], person_id="alice", chat_id="chat", message_ids=["new"]
        )
        assert store.supersede_person_fact_from_evidence(**kwargs)
        assert store.get_fact_claim(old["claim_id"])["status"] == "superseded"
        assert store.project_fact_paragraph(store.get_paragraph(old_paragraph)) is None
        assert not store.supersede_person_fact_from_evidence(**kwargs)
        store.retract_fact_claim(new["claim_id"], reason="user removed")
        assert not store.supersede_person_fact_from_evidence(**kwargs)
        assert store.get_fact_claim(new["claim_id"])["status"] == "retracted"
    finally:
        store.close()


def test_correction_rejects_cross_chat_manual_and_stale_versions(tmp_path):
    import pytest
    from src.A_memorix.core.storage.metadata_store import MetadataStore

    store = MetadataStore(data_dir=tmp_path)
    store.connect()
    try:
        old, _ = _claim(store, "Alice 喜欢喝茶", "old")
        new, _ = _claim(store, "Alice 不再喝茶", "new", chat_id="elsewhere")
        kwargs = dict(
            previous=old, new_claim_id=new["claim_id"], person_id="alice", chat_id="chat", message_ids=["new"]
        )
        with pytest.raises(ValueError, match="同群"):
            store.supersede_person_fact_from_evidence(**kwargs)
        new, _ = _claim(store, "Alice 现在喝咖啡", "new2")
        kwargs.update(new_claim_id=new["claim_id"], message_ids=["new2"])
        with pytest.raises(ValueError, match="已变化"):
            store.supersede_person_fact_from_evidence(**{**kwargs, "previous": {**old, "updated_at": -1}})
        manual, _ = _claim(store, "Alice 喜欢猫", "manual", authority="manual")
        with pytest.raises(ValueError, match="人工"):
            store.supersede_person_fact_from_evidence(**{**kwargs, "previous": manual})
        assert store.get_fact_claim(old["claim_id"])["status"] == "active"
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_transition", [False, True])
async def test_correction_crash_window_replays_ingest_and_transition_once(tmp_path, monkeypatch, after_transition):
    from src.A_memorix.core.runtime.services.ingest_service import MemoryIngestService
    from src.A_memorix.core.storage.metadata_store import MetadataStore

    store = MetadataStore(data_dir=tmp_path)
    store.connect()
    old, _ = _claim(store, "Alice 喜欢喝茶", "old")
    saved = {}
    writes = []

    async def initialize():
        pass

    async def ingest(kwargs, metadata, person_id):
        if not saved:
            new, paragraph = _claim(store, "Alice 不再喝茶", "new")
            saved.update(success=True, stored_ids=[paragraph], fact_claim_ids=[new["claim_id"]])
            writes.append(new["claim_id"])
        return saved

    kernel = NS(
        metadata_store=store,
        initialize=initialize,
        _is_chat_filtered=lambda **kwargs: False,
        _ingest_person_fact=ingest,
    )
    service = MemoryIngestService(kernel)
    original = store.supersede_person_fact_from_evidence
    attempts = []

    def interrupted(**kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            if after_transition:
                original(**kwargs)
            raise RuntimeError("simulated process stop")
        return original(**kwargs)

    monkeypatch.setattr(store, "supersede_person_fact_from_evidence", interrupted)
    kwargs = dict(
        source_type="person_fact",
        person_ids=["alice"],
        chat_id="chat",
        text="Alice 不再喝茶",
        external_id="receipt",
        metadata={
            "writeback_source": "memory_flow_service",
            "evidence_message_ids": ["new"],
            "replaces_person_fact": old,
        },
    )
    try:
        with pytest.raises(RuntimeError, match="process stop"):
            await service.ingest_text(**kwargs)
        result = await service.ingest_text(**kwargs)
        assert result["success"]
        assert len(writes) == 1
        assert store.get_fact_claim(old["claim_id"])["status"] == "superseded"
        transitions = store.query("SELECT * FROM fact_transitions WHERE transition_type='supersede'")
        assert len(transitions) == 1
    finally:
        store.close()


def test_correction_can_reuse_existing_claim_with_new_source_evidence(tmp_path):
    from src.A_memorix.core.storage.metadata_store import MetadataStore
    from src.A_memorix.core.utils.fact_deduplication import FactDeduplicator

    store = MetadataStore(data_dir=tmp_path)
    store.connect()
    try:
        old, _ = _claim(store, "Alice 住上海", "old")
        new, paragraph = _claim(store, "Alice 住北京", "earlier")
        plan = {
            "scope_type": "person",
            "scope_id": "alice",
            "facts": [{"fact_id": "f1", "text": new["value_text"]}],
            "source": {"chat_id": "chat", "message_ids": ["correction"], "observation_id": "observation-correction"},
            "decisions": [
                {
                    "fact_id": "f1",
                    "action": "reuse",
                    "target": {"claim_id": new["claim_id"], "paragraph_hash": paragraph},
                }
            ],
        }
        assert FactDeduplicator(store).commit(plan) == [new["claim_id"]]
        assert store.supersede_person_fact_from_evidence(
            previous=old, new_claim_id=new["claim_id"], person_id="alice", chat_id="chat", message_ids=["correction"]
        )
        assert store.get_fact_claim(old["claim_id"])["status"] == "superseded"
    finally:
        store.close()
