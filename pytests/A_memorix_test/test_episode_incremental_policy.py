from typing import Any

import time

import pytest

from src.A_memorix.core.storage.metadata_store import MetadataStore
from src.A_memorix.core.utils.episode_service import EpisodeService


class Segmenter:
    def __init__(self, version: str = "v1", fail_at: int = 0, empty: bool = False):
        self.version, self.fail_at, self.empty = version, fail_at, empty
        self.calls = []

    def generation_signature(self):
        return {"version": self.version}

    async def segment(self, **kwargs: Any):
        ids = [p["hash"] for p in kwargs["paragraphs"]]
        self.calls.append(ids)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("interrupted")
        return {
            "episodes": [] if self.empty else [{"title": self.version, "summary": "event", "paragraph_hashes": ids}],
            "segmentation_model": "test",
            "segmentation_version": self.version,
        }


def service(store, segmenter):
    return EpisodeService(
        metadata_store=store, segmentation_service=segmenter, plugin_config={"episode": {"max_paragraphs_per_call": 2}}
    )


@pytest.fixture
def store(tmp_path):
    s = MetadataStore(data_dir=tmp_path)
    s.connect()
    yield s
    s.close()


def add(store, n=4):
    return [
        store.add_paragraph(f"paragraph {i}", source="chat_summary:test", time_meta={"event_time": i}) for i in range(n)
    ]


@pytest.mark.asyncio
async def test_version_change_keeps_clean_source_and_old_queries(store):
    add(store)
    s = service(store, Segmenter())
    await s.rebuild_source("chat_summary:test")
    claim = store.claim_episode_source_rebuild_batch(
        generation_hash=s.generation_hash(), max_wait_seconds=0, now=time.time() + 5
    )[0]
    store.publish_episode_source_rebuild(
        "chat_summary:test",
        lease_token=claim["lease_token"],
        claimed_revision=claim["claimed_revision"],
        generation_hash=s.generation_hash(),
        episodes_payloads=store.get_episodes_by_source("chat_summary:test"),
    )
    changed = service(store, Segmenter("v2"))
    assert (
        store.claim_episode_source_rebuild_batch(
            generation_hash=changed.generation_hash(), max_wait_seconds=0, now=time.time() + 10
        )
        == []
    )
    assert len(store.query_episodes(source="chat_summary:test")) == 2


@pytest.mark.asyncio
async def test_version_change_only_new_material_uses_new_rules(store):
    add(store)
    await service(store, Segmenter()).rebuild_source("chat_summary:test")
    store.add_paragraph("new material", source="chat_summary:test", time_meta={"event_time": 5})
    model = Segmenter("v2")
    result = await service(store, model).rebuild_source("chat_summary:test")
    assert len(model.calls) == 1
    assert result["reused_group_count"] == 2
    assert sorted(e["title"] for e in store.get_episodes_by_source("chat_summary:test")) == ["v1", "v1", "v2"]


@pytest.mark.asyncio
async def test_interrupted_source_reuses_completed_groups_after_restart(store):
    add(store)
    with pytest.raises(RuntimeError, match="interrupted"):
        await service(store, Segmenter(fail_at=2)).plan_source_rebuild("chat_summary:test")
    model = Segmenter()
    await service(store, model).rebuild_source("chat_summary:test")
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_empty_group_is_checkpointed_before_later_failure(store):
    add(store)
    with pytest.raises(RuntimeError, match="interrupted"):
        await service(store, Segmenter(fail_at=2, empty=True)).plan_source_rebuild("chat_summary:test")
    model = Segmenter(empty=True)
    await service(store, model).rebuild_source("chat_summary:test")
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_correction_only_recomputes_affected_group(store):
    ids = add(store, 6)
    await service(store, Segmenter()).rebuild_source("chat_summary:test")
    store.update_paragraph_metadata(ids[0], {"summary_event_spans": ["corrected event"]})
    model = Segmenter("v2")
    await service(store, model).rebuild_source("chat_summary:test")
    assert model.calls == [ids[:2]]
    assert len(store.get_episodes_by_source("chat_summary:test")) == 3


@pytest.mark.asyncio
async def test_delete_does_not_shift_or_recompute_unaffected_groups(store):
    ids = add(store, 6)
    await service(store, Segmenter()).rebuild_source("chat_summary:test")
    store.mark_as_deleted([ids[0]], "paragraph", reason="correction")
    model = Segmenter("v2")
    await service(store, model).rebuild_source("chat_summary:test")
    assert model.calls == [[ids[1]]]
    assert ids[0] not in [h for e in store.get_episodes_by_source("chat_summary:test") for h in e["evidence_ids"]]


@pytest.mark.asyncio
async def test_first_upgrade_preserves_legacy_published_groups(store):
    from src.A_memorix.core.utils.episode_group_cache import EpisodeGroupCache

    add(store)
    await service(store, Segmenter()).rebuild_source("chat_summary:test")
    EpisodeGroupCache(store.get_db_path().with_name("episode_groups.sqlite3")).retain("chat_summary:test", [])
    store.add_paragraph("new tail", source="chat_summary:test", time_meta={"event_time": 7})
    model = Segmenter("v2")
    await service(store, model).rebuild_source("chat_summary:test")
    assert len(model.calls) == 1
    assert sorted(e["title"] for e in store.get_episodes_by_source("chat_summary:test")) == ["v1", "v1", "v2"]


@pytest.mark.asyncio
async def test_call_budget_stops_before_second_model_and_keeps_first_result(store):
    from src.A_memorix.core.utils.episode_group_cache import EpisodeCallBudget

    add(store)
    model = Segmenter("v2")
    s = service(store, model)
    budget = EpisodeCallBudget(1)
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        await s.plan_source_rebuild("chat_summary:test", upgrade_generation=True, call_budget=budget)
    assert len(model.calls) == budget.used == 1
    remaining = EpisodeCallBudget(1)
    await s.plan_source_rebuild("chat_summary:test", upgrade_generation=True, call_budget=remaining)
    assert remaining.used == 1
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_manual_upgrade_preview_and_budget_are_required(store, monkeypatch, tmp_path):
    from src.A_memorix.core.runtime.sdk_memory_kernel import SDKMemoryKernel

    add(store)
    kernel = SDKMemoryKernel(plugin_root=tmp_path, config={})
    kernel.metadata_store = store
    model = Segmenter()
    kernel.episode_service = service(store, model)

    async def initialize():
        return None

    monkeypatch.setattr(kernel, "initialize", initialize)
    monkeypatch.setattr(kernel, "_persist", lambda: None)
    preview = (await kernel.memory_episode_admin(action="preview_rebuild", sources=["chat_summary:test"]))["preview"]
    assert preview["paragraph_count"] == 4 and preview["model_calls_upper_bound"] == 2
    assert not model.calls
    missing = await kernel.memory_episode_admin(action="rebuild", sources=["chat_summary:test"])
    assert missing["success"] is False
    too_small = await kernel.memory_episode_admin(
        action="rebuild", sources=["chat_summary:test"], preview_id=preview["preview_id"], max_model_calls=1
    )
    assert too_small["success"] is False and not model.calls
    done = await kernel.memory_episode_admin(
        action="rebuild", sources=["chat_summary:test"], preview_id=preview["preview_id"], max_model_calls=2
    )
    assert done["success"] is True and done["model_calls_used"] == 2
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_stale_preview_does_not_start_or_enqueue_upgrade(store, monkeypatch, tmp_path):
    from src.A_memorix.core.runtime.sdk_memory_kernel import SDKMemoryKernel

    add(store)
    kernel = SDKMemoryKernel(plugin_root=tmp_path, config={})
    kernel.metadata_store = store
    model = Segmenter()
    kernel.episode_service = service(store, model)

    async def initialize():
        return None

    monkeypatch.setattr(kernel, "initialize", initialize)
    preview = (await kernel.memory_episode_admin(action="preview_rebuild", sources=["chat_summary:test"]))["preview"]
    store.add_paragraph("changed", source="chat_summary:test")
    result = await kernel.memory_episode_admin(
        action="rebuild", sources=["chat_summary:test"], preview_id=preview["preview_id"], max_model_calls=20
    )
    assert result["success"] is False
    assert result["preview"]["preview_id"] != preview["preview_id"]
    assert not model.calls
    assert store.list_episode_source_rebuilds()[0]["reason"] != "episode_admin_rebuild"


def test_background_worker_does_not_take_over_manual_budget(store):
    store.enqueue_episode_source_rebuild("chat_summary:test", reason="episode_admin_rebuild", debounce_seconds=0)
    assert store.claim_episode_source_rebuild_batch(generation_hash="v2", now=time.time() + 10) == []
    assert store.claim_episode_source_rebuild_batch(
        generation_hash="v2", sources=["chat_summary:test"], now=time.time() + 10
    )
