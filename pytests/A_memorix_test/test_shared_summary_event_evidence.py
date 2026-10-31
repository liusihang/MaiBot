import pytest

from src.A_memorix.core.utils.episode_segmentation_service import EpisodeSegmentationService
from src.A_memorix.core.utils.summary_event_evidence import validated_event_spans


def test_exact_event_partitions_reject_omissions_rewrites_and_overlap():
    assert validated_event_spans(
        ["Alice finished tests.", "Bob sent the report."], "Alice finished tests.\nBob sent the report."
    )
    assert not validated_event_spans(["Alice finished tests."], "Alice finished tests. Bob sent the report.")
    assert not validated_event_spans(["Alice passed tests."], "Alice finished tests.")
    assert not validated_event_spans(
        ["Bob sent the report.", "Alice finished tests."], "Alice finished tests. Bob sent the report."
    )
    assert not validated_event_spans(["Alice finished tests.", "Alice finished tests."], "Alice finished tests.")


@pytest.mark.asyncio
async def test_single_window_multiple_validated_events_needs_no_second_model():
    service = EpisodeSegmentationService()
    service._resolve_model_config = lambda: (_ for _ in ()).throw(AssertionError("redundant model resolution"))
    content = "Alice finished tests.\nBob sent the report."
    paragraph = {
        "hash": "p1",
        "content": content,
        "metadata": {
            "source_window_id": "window-1",
            "chat_id": "chat-1",
            "summary_event_spans": ["Alice finished tests.", "Bob sent the report."],
        },
    }
    result = await service.segment(source="chat_summary:chat-1", window_start=1, window_end=2, paragraphs=[paragraph])
    assert result["segmentation_model"] == "reused_summary_boundaries"
    assert len(result["episodes"]) == 2
    assert all(event["paragraph_hashes"] == ["p1"] for event in result["episodes"])


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["rewritten", "other_chat", "missing_window", "multiple_windows"])
async def test_unsafe_or_cross_window_reuse_retains_normal_segmentation(case):
    service = EpisodeSegmentationService()
    service._resolve_model_config = lambda: (None, "missing")
    paragraph = {
        "hash": "p1",
        "content": "Event.",
        "metadata": {"source_window_id": "w", "chat_id": "chat-1", "summary_event_spans": ["Event."]},
    }
    if case == "rewritten":
        paragraph["content"] = "Different event."
    elif case == "other_chat":
        paragraph["metadata"]["chat_id"] = "other"
    elif case == "missing_window":
        paragraph["metadata"].pop("source_window_id")
    paragraphs = [paragraph, paragraph] if case == "multiple_windows" else [paragraph]
    with pytest.raises(RuntimeError, match="model unavailable"):
        await service.segment(source="chat_summary:chat-1", window_start=1, window_end=2, paragraphs=paragraphs)


@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected", [("Event.", ["Event."]), ("Deduplicated fact.", [])])
async def test_import_revalidates_boundaries_after_text_changes(text, expected):
    from types import SimpleNamespace
    from src.A_memorix.core.utils.summary_importer import SummaryImporter

    captured = {}

    async def ingest(**kwargs):
        captured.update(kwargs)
        return {"stored_ids": ["p1"]}

    importer = object.__new__(SummaryImporter)
    importer._plugin_instance = lambda: SimpleNamespace(ingest_text=ingest)
    importer.metadata_store = SimpleNamespace(get_external_memory_ref=lambda _: {"paragraph_hash": "p1"})
    await importer._execute_import(
        text, [], [], "chat-1", metadata={"external_id": "window-1", "summary_event_spans": ["Event."]}
    )
    assert captured["metadata"]["summary_event_spans"] == expected


def test_changed_summary_boundaries_invalidate_episode_group_cache():
    from src.A_memorix.core.utils.episode_service import EpisodeService

    service = EpisodeService(metadata_store=None)
    paragraph = {
        "hash": "p1",
        "content": "One. Two.",
        "metadata": {"source_window_id": "w", "chat_id": "c", "summary_event_spans": ["One. Two."]},
    }
    group = {"source": "chat_summary:c", "paragraphs": [paragraph], "_segmentation_generation": {"version": "test"}}
    before = service._group_input_fingerprint(group)
    paragraph["metadata"]["summary_event_spans"] = ["One.", "Two."]
    assert service._group_input_fingerprint(group) != before


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", [["m1"], ["m1", "m1"], ["other", "m2"], "m1"])
async def test_summary_window_must_match_original_message_ids(monkeypatch, expected):
    from types import SimpleNamespace
    from src.A_memorix.core.utils import summary_importer as module

    importer = object.__new__(module.SummaryImporter)
    importer._existing_summary_result = lambda **_: None
    importer.plugin_config = {}

    async def ready():
        return True, "ready"

    importer._ensure_runtime_self_check = ready
    monkeypatch.setattr(
        module.message_api,
        "get_messages_by_time_in_chat",
        lambda **_: [SimpleNamespace(message_id="m1"), SimpleNamespace(message_id="m2")],
    )
    result = await importer._import_from_stream_scoped("chat-1", metadata={"source_message_ids": expected})
    assert result.success is False
    assert "来源窗口已变化" in result.detail
