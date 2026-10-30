from pathlib import Path

import pytest

from src.maisaka.pi_runtime.business_store import BusinessStore, LedgerConflict


def test_inbox_is_business_delivery_not_second_transcript(tmp_path: Path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    assert store.enqueue_input("group", "source-1", {"text": "hello"})
    assert not store.enqueue_input("group", "source-1", {"text": "hello"})
    assert [item.event_id for item in store.pending_inputs("group")] == ["source-1"]
    assert store.pending_inputs("other") == []
    store.acknowledge_input("group", "source-1")
    restored = BusinessStore(tmp_path / "business.sqlite3")
    assert restored.has_input("group", "source-1")
    assert restored.pending_inputs("group") == []
    with pytest.raises(LedgerConflict):
        restored.enqueue_input("group", "source-1", {"text": "changed"})
    assert not hasattr(store, "snapshot")
    assert not hasattr(store, "append_event")
    assert not hasattr(store, "commit_compaction")


def test_unknown_side_effect_fences_new_calls_and_restart(tmp_path: Path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    store.prepare_tool_execution("group", "call", {"name": "reply", "args": {"text": "hi"}})
    assert store.claim_tool_execution("group", "call")
    store.mark_tool_execution_unknown("group", "call", observation={"sent_message_ids": ["receipt-1"]})
    restored = BusinessStore(tmp_path / "business.sqlite3")
    with pytest.raises(LedgerConflict):
        restored.assert_no_unresolved_executions("group")
    restored.prepare_tool_execution("group", "new-call", {"name": "reply"})
    assert not restored.claim_tool_execution("group", "new-call")
    assert restored.lookup_tool_execution("group", "call").result == {"sent_message_ids": ["receipt-1"]}
    restored.assert_no_unresolved_executions("other")


def test_known_receipt_can_recover_without_redispatch(tmp_path: Path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    intent = {"name": "reply", "arguments": {"text": "hi"}}
    store.prepare_tool_execution("group", "call", intent)
    assert store.claim_tool_execution("group", "call")
    result = {"content": [{"type": "text", "text": "sent"}], "isError": False}
    store.record_tool_outcome("group", "call", result)
    restored = BusinessStore(tmp_path / "business.sqlite3")
    assert restored.prepare_tool_execution("group", "call", intent).result == result
    assert not restored.claim_tool_execution("group", "call")
    restored.assert_no_unresolved_executions("group")
    with pytest.raises(LedgerConflict):
        restored.prepare_tool_execution("group", "call", {"name": "different"})


def test_original_source_fingerprint_stays_scoped_and_durable_after_inbox_ack(tmp_path: Path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    store.validate_source("group-a", "source-1", "raw-fingerprint")
    store.enqueue_input("group-a", "source-1", {"text": "first frozen rendering"})
    store.acknowledge_input("group-a", "source-1")
    reopened = BusinessStore(tmp_path / "business.sqlite3")
    reopened.validate_source("group-a", "source-1", "raw-fingerprint")
    assert reopened.pending_inputs("group-a") == []
    with pytest.raises(LedgerConflict, match="original content"):
        reopened.validate_source("group-a", "source-1", "changed-raw")
    reopened.validate_source("group-b", "source-1", "different-group-raw")
    assert reopened.has_input("group-a", "source-1")
    assert not reopened.has_input("group-b", "source-1")


def test_tool_media_ownership_commits_with_receipt_and_cannot_rebind(tmp_path: Path):
    path = tmp_path / "business.sqlite3"
    store = BusinessStore(path)
    store.prepare_tool_execution("group", "call-1", {"name": "show_emoji_list"})
    assert store.claim_tool_execution("group", "call-1")
    result = {
        "content": [{"type": "text", "text": "collage"}],
        "isError": False,
        "details": {"hostMediaEventIds": ["input:item:collage-1"]},
    }
    store.record_tool_outcome("group", "call-1", result)
    restored = BusinessStore(path)
    assert restored.is_tool_media_input("group", "input:item:collage-1")
    assert not restored.is_tool_media_input("other", "input:item:collage-1")
    assert restored.lookup_tool_execution("group", "call-1").result == result
    restored.prepare_tool_execution("group", "call-2", {"name": "show_emoji_list"})
    assert restored.claim_tool_execution("group", "call-2")
    with pytest.raises(LedgerConflict, match="rebound"):
        restored.record_tool_outcome("group", "call-2", result)
    assert restored.lookup_tool_execution("group", "call-2").result is None
    assert restored.lookup_tool_execution("group", "call-1").result == result


def test_selected_turn_survives_inbox_ack_and_completion_is_scoped(tmp_path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    store.enqueue_input("ignored", "a", {"sourceMessageId": "a"})
    store.acknowledge_input("ignored", "a")
    turn = store.begin_turn("group", ["one"])
    assert store.begin_turn("group", ["two"])["turn_id"] == turn["turn_id"]
    restored = BusinessStore(tmp_path / "business.sqlite3")
    assert restored.pending_turn("group")["source_ids"] == ["one", "two"]
    assert restored.recovery_streams() == ["group"]
    with pytest.raises(LedgerConflict):
        restored.complete_turn("group", "other-turn")
    restored.complete_turn("group", turn["turn_id"])
    assert restored.pending_turn("group") is None
    assert restored.recovery_streams() == []
