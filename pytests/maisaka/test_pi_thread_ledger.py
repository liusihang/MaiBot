"""Offline contracts for canonical thread storage; no models or external effects."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import asyncio
import json
import sqlite3

import pytest

from src.maisaka.pi_runtime.ledger import EventInput, LedgerConflict, ThreadLedger, UnsafeCompaction


@pytest.fixture
def ledger(tmp_path):
    return ThreadLedger(tmp_path / "threads.sqlite3")


def native_call(call_id="call-1"):
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "native reasoning", "thinkingSignature": "opaque signature"},
            {"type": "toolCall", "id": call_id, "name": "reply", "arguments": {"text": "hello"}},
        ],
        "stopReason": "toolUse",
        "usage": {"input": 31, "output": 5},
    }


def native_result(call_id="call-1"):
    return {
        "role": "toolResult",
        "toolCallId": call_id,
        "toolName": "reply",
        "content": [{"type": "text", "text": "sent"}],
        "isError": False,
        "details": {"readReceipt": {"messageId": "ack-1"}},
    }


def test_native_payload_roundtrip_no_mutation_and_reopen(ledger):
    payload = native_call()
    payload["providerMetadata"] = {"unknown": [None, False, 1, 1.5, "中文", {"x": "y"}]}
    snapshot = ledger.append_event("real-session-a", "source-1", payload, expected_revision=0)
    assert snapshot.revision == 1
    assert snapshot.events[0].payload == payload
    assert json.loads(snapshot.events[0].payload_json) == payload
    original = snapshot.events[0].payload_json
    payload["content"].clear()
    snapshot.events[0].payload["providerMetadata"].clear()
    assert ledger.snapshot("real-session-a").events[0].payload_json == original
    assert ThreadLedger(ledger.db_path).snapshot("real-session-a") == snapshot
    assert ledger.snapshot("other-session").events == ()


def test_batch_idempotency_and_atomic_compare_and_swap(ledger):
    events = [EventInput("m-1", {"role": "user", "content": "one"}), EventInput("m-2", {"content": "two"})]
    first = ledger.append_events("a", events, expected_revision=0)
    assert first.revision == 1
    assert ledger.append_events("a", events) == first
    with pytest.raises(LedgerConflict, match="revision"):
        ledger.append_event("a", "m-3", {}, expected_revision=0)
    with pytest.raises(LedgerConflict, match="immutable event"):
        ledger.append_events("a", [EventInput("m-3", {}), EventInput("m-1", {"changed": True})])
    assert ledger.snapshot("a") == first
    # Source ids are scoped to the actual host stream, not globally.
    assert ledger.append_event("b", "m-1", {"different": True}).events[0].payload == {"different": True}
    with pytest.raises(KeyError):
        ledger.archive_events("b", ["m-2"])


@pytest.mark.parametrize("payload", [float("nan"), {1: "not a string key"}, {"tuple": (1, 2)}, {"bytes": b"x"}])
def test_reject_lossy_non_json_payloads(ledger, payload):
    with pytest.raises(TypeError):
        ledger.append_event("a", "bad", payload)
    assert ledger.snapshot("a").revision == 0


def test_concurrent_idempotent_appends_and_cas(tmp_path):
    path = tmp_path / "shared.sqlite3"
    ledger = ThreadLedger(path)

    def append(index):
        ThreadLedger(path).append_event("a", f"message-{index % 20}", {"index": index % 20})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(60)))
    snapshot = ledger.snapshot("a")
    assert snapshot.revision == 20
    assert len(snapshot.events) == 20
    assert len({event.event_id for event in snapshot.events}) == 20

    def cas(index):
        try:
            ledger.append_event("a", f"cas-{index}", {}, expected_revision=snapshot.revision)
            return True
        except LedgerConflict:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(cas, range(4))) == 1


def test_archive_is_append_only_even_for_direct_sql(ledger):
    ledger.append_event("a", "m-1", {"text": "original"})
    with sqlite3.connect(ledger.db_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE thread_events SET payload_json = '{}' WHERE event_id = 'm-1'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM thread_events WHERE event_id = 'm-1'")


def test_reference_version_delta_and_no_deletion_on_nonmatches(ledger):
    first = ledger.append_reference("a", "memory", "memory-1", "v1", {"text": "original"})
    assert first.appended and first.event.payload["authority"] == "reference"
    repeat = ledger.append_reference("a", "memory", "memory-1", "v1", {"text": "original"})
    assert not repeat.appended and repeat.snapshot == first.snapshot
    updated = ledger.append_reference("a", "memory", "memory-1", "v2", {"text": "updated"})
    assert updated.appended
    assert updated.event.payload["supersedes"] == first.event.event_id
    assert updated.snapshot.events[0] == first.event  # No mutation/replacement of old reference.
    ledger.append_event("a", "unrelated-message", {"content": "no reference matched here"})
    assert ledger.snapshot("a").events[:2] == updated.snapshot.events
    with pytest.raises(LedgerConflict, match="new source version"):
        ledger.append_reference("a", "memory", "memory-1", "v2", {"text": "changed without version"})


def test_reference_identity_is_per_item_namespace_and_stream(ledger):
    for namespace, source in (("memory", "M1"), ("memory", "M2"), ("person", "M1"), ("image", "M1")):
        assert ledger.append_reference("a", namespace, source, "1", "same content").appended
    assert len(ledger.snapshot("a").events) == 4
    assert not ledger.append_reference("a", "memory", "M1", "1", "same content").appended
    assert ledger.append_reference("b", "memory", "M1", "1", "same content").appended


@pytest.mark.parametrize("source_id,version", [(None, "1"), ("known-id", None), (None, None)])
def test_unresolved_identity_requires_explicit_full_content_fingerprint(ledger, source_id, version):
    with pytest.raises(ValueError, match="fallback_scope"):
        ledger.append_reference("a", "jargon", source_id, version, "x")
    content = "not truncated " * 1000
    first = ledger.append_reference("a", "jargon", source_id, version, content, fallback_scope="matched-record")
    identity = first.event.payload["identity"]
    assert identity["kind"] == "fingerprint" and identity["stable"] is False
    assert "unavailable" in identity["disclosure"]
    assert not ledger.append_reference(
        "a", "jargon", source_id, version, content, fallback_scope="matched-record"
    ).appended
    assert ledger.append_reference(
        "a", "jargon", source_id, version, content + "last character changed", fallback_scope="matched-record"
    ).appended
    assert ledger.append_reference(
        "a", "jargon", source_id, version, content, fallback_scope="different-record"
    ).appended


def test_explicit_read_is_never_suppressed(ledger):
    first = ledger.append_reference("a", "person", "user-1", "v1", {"facts": ["x"]})
    read = ledger.append_reference("a", "person", "user-1", "v1", {"facts": ["x"]}, force=True)
    assert read.appended and read.reference_key == first.reference_key
    assert read.event.event_id != first.event.event_id
    assert len(read.snapshot.events) == 2


def test_compaction_preserves_concurrent_suffix_and_exact_original_archive(ledger):
    original = ledger.append_events(
        "a", [EventInput(f"m-{index}", {"text": f"original {index}"}) for index in range(4)]
    )
    plan = ledger.begin_compaction("a", 2)
    ledger.append_event("a", "new-during-summary", {"text": "new arrival"})
    result = ledger.commit_compaction(plan, {"summary": "old first two", "opaque": [1, None]})
    assert result.events[0].kind == "summary"
    assert [event.event_id for event in result.events[1:]] == ["m-2", "m-3", "new-during-summary"]
    assert ledger.archive_events("a", ["m-0", "m-1"]) == original.events[:2]
    assert (
        ThreadLedger(ledger.db_path).commit_compaction(plan, {"summary": "old first two", "opaque": [1, None]})
        == result
    )
    with pytest.raises(LedgerConflict, match="token"):
        ledger.commit_compaction(plan, {"different summary": True})
    # Delivery replays do not resurrect the compacted original message.
    assert ledger.append_event("a", "m-0", {"text": "original 0"}) == result


def test_compaction_conflicting_prefix_or_modified_plan_is_rejected(ledger):
    before = ledger.append_events("a", [EventInput(f"m-{index}", {"text": str(index)}) for index in range(3)])
    first_plan = ledger.begin_compaction("a", 1)
    competing_plan = ledger.begin_compaction("a", 2)
    changed = replace(first_plan, prefix=(replace(first_plan.prefix[0], payload_json='{"text":"tampered"}'),))
    with pytest.raises(LedgerConflict, match="prefix changed"):
        ledger.commit_compaction(changed, {})
    assert ledger.snapshot("a") == before
    after = ledger.commit_compaction(first_plan, {"summary": "one"})
    with pytest.raises(LedgerConflict, match="prefix changed"):
        ledger.commit_compaction(competing_plan, {"summary": "stale two"})
    assert ledger.snapshot("a") == after


@pytest.mark.asyncio
async def test_failed_or_cancelled_summary_leaves_head_unchanged(ledger):
    before = await asyncio.to_thread(ledger.append_event, "a", "m-1", {"text": "keep me"})

    async def failed_summary(error):
        await asyncio.to_thread(ledger.begin_compaction, "a", 1)
        raise error

    for error in (RuntimeError("provider failed"), asyncio.CancelledError()):
        with pytest.raises(type(error)):
            await failed_summary(error)
        assert await asyncio.to_thread(ledger.snapshot, "a") == before


def test_compaction_releases_reference_unless_summary_explicitly_retains_it(ledger):
    original = ledger.append_reference("a", "memory", "M1", "v1", "important fact")
    first_plan = ledger.begin_compaction("a", 1)
    kept = ledger.commit_compaction(
        first_plan, {"summary": "important fact"}, retained_reference_keys=[original.reference_key]
    )
    assert not ledger.append_reference("a", "memory", "M1", "v1", "important fact").appended
    second_plan = ledger.begin_compaction("a", 1)
    ledger.commit_compaction(second_plan, {"summary": "fact no longer visible"})
    reinjected = ledger.append_reference("a", "memory", "M1", "v1", "important fact")
    assert reinjected.appended and reinjected.event.event_id != original.event.event_id
    assert ledger.archive_events("a", [original.event.event_id])[0] == original.event
    assert ledger.archive_events("a", [kept.events[0].event_id])[0] == kept.events[0]


def test_compaction_cannot_invent_retained_reference_keys(ledger):
    before = ledger.append_event("a", "m-1", {})
    plan = ledger.begin_compaction("a", 1)
    with pytest.raises(ValueError, match="absent"):
        ledger.commit_compaction(plan, {}, retained_reference_keys=["invented-reference"])
    assert ledger.snapshot("a") == before


def test_safe_prefix_cannot_split_native_tool_calls_or_results(ledger):
    ledger.append_event("a", "user", {"role": "user", "content": "hello"})
    ledger.append_event("a", "assistant", native_call())
    assert ledger.begin_compaction("a", 1).prefix[0].event_id == "user"
    with pytest.raises(UnsafeCompaction, match="before tool results"):
        ledger.begin_compaction("a", 2)
    ledger.append_event("a", "tool-result", native_result())
    with pytest.raises(UnsafeCompaction):
        ledger.begin_compaction("a", 2)
    assert len(ledger.begin_compaction("a", 3).prefix) == 3
    ledger.append_event("b", "orphan-result", native_result())
    with pytest.raises(UnsafeCompaction, match="no matching"):
        ledger.begin_compaction("b", 1)


def test_explicit_tool_metadata_can_protect_other_native_shapes(ledger):
    ledger.append_event("a", "custom-call", {"customShape": ["opaque"]}, tool_call_ids=("custom",))
    with pytest.raises(UnsafeCompaction):
        ledger.begin_compaction("a", 1)
    ledger.append_event("a", "custom-result", {"readReceipt": "opaque"}, tool_result_ids=("custom",))
    assert len(ledger.begin_compaction("a", 2).prefix) == 2


def test_execution_intent_is_idempotent_and_only_one_concurrent_claim(ledger):
    intent = {"name": "reply", "arguments": {"text": "hello"}}
    receipt = ledger.prepare_tool_execution("a", "call-1", intent)
    assert receipt.status == "prepared" and receipt.intent == intent
    assert ledger.prepare_tool_execution("a", "call-1", intent) == receipt
    with pytest.raises(LedgerConflict, match="different execution intent"):
        ledger.prepare_tool_execution("a", "call-1", {"text": "changed effect"})
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: ledger.claim_tool_execution("a", "call-1"), range(16))) == 1
    restarted = ThreadLedger(ledger.db_path)
    assert restarted.lookup_tool_execution("a", "call-1").status == "executing"
    assert restarted.claim_tool_execution("a", "call-1") is False  # ACK uncertain; no retry.
    assert restarted.lookup_tool_execution("b", "call-1") is None
    assert restarted.claim_tool_execution("a", "missing-call") is False


def test_unknown_execution_never_retries_but_authoritative_result_can_reconcile(ledger):
    ledger.append_event("a", "assistant", native_call())
    ledger.prepare_tool_execution("a", "call-1", {"name": "reply"})
    assert ledger.claim_tool_execution("a", "call-1")
    receipt = ledger.mark_tool_execution_unknown("a", "call-1")
    assert receipt.status == "unknown"
    assert ledger.mark_tool_execution_unknown("a", "call-1") == receipt
    assert not ledger.claim_tool_execution("a", "call-1")
    with pytest.raises(UnsafeCompaction):
        ledger.begin_compaction("a", 1)
    completed = ledger.complete_tool_execution("a", "call-1", "ack-event", native_result())
    assert completed.events[-1].payload == native_result()
    receipt = ledger.lookup_tool_execution("a", "call-1")
    assert (
        receipt.status == "succeeded" and receipt.result == native_result() and receipt.result_event_id == "ack-event"
    )
    assert not ledger.claim_tool_execution("a", "call-1")
    assert ledger.complete_tool_execution("a", "call-1", "ack-event", native_result()) == completed
    assert len(ledger.begin_compaction("a", 2).prefix) == 2
    with pytest.raises(LedgerConflict, match="immutable"):
        ledger.complete_tool_execution("a", "call-1", "ack-event", {"changed": True})
    with pytest.raises(LedgerConflict):
        ledger.mark_tool_execution_unknown("a", "call-1")


def test_execution_result_and_receipt_commit_atomically(ledger):
    ledger.prepare_tool_execution("a", "call-1", {"name": "reply"})
    with pytest.raises(LedgerConflict, match="claimed"):
        ledger.complete_tool_execution("a", "call-1", "result", native_result())
    ledger.claim_tool_execution("a", "call-1")
    before = ledger.append_event("a", "result", {"different immutable message": True})
    with pytest.raises(LedgerConflict, match="immutable event"):
        ledger.complete_tool_execution("a", "call-1", "result", native_result())
    assert ledger.snapshot("a") == before
    assert ledger.lookup_tool_execution("a", "call-1").status == "executing"
    error = {**native_result(), "isError": True}
    ledger.complete_tool_execution("a", "call-1", "error-result", error, status="failed")
    assert ledger.lookup_tool_execution("a", "call-1").status == "failed"
    assert not ledger.claim_tool_execution("a", "call-1")


def test_unresolved_receipt_blocks_compaction_even_with_manually_appended_result(ledger):
    ledger.append_event("a", "assistant", native_call())
    ledger.prepare_tool_execution("a", "call-1", {"name": "reply"})
    ledger.claim_tool_execution("a", "call-1")
    ledger.mark_tool_execution_unknown("a", "call-1")
    ledger.append_event("a", "result", native_result())
    with pytest.raises(UnsafeCompaction, match="unresolved execution"):
        ledger.begin_compaction("a", 2)
    ledger.complete_tool_execution("a", "call-1", "result", native_result())
    assert len(ledger.begin_compaction("a", 2).prefix) == 2


def test_compaction_preserves_fixed_system_declarations_and_source_dedup(ledger):
    system = {"role": "system", "content": "unchanged system", "toolsAdded": [{"name": "read", "parameters": {}}]}
    ledger.append_event("a", "system", system)
    ledger.append_event("a", "m-1", {"role": "user", "content": "old message"})
    with pytest.raises(UnsafeCompaction, match="stable system"):
        ledger.begin_compaction("a", 2)
    plan = ledger.begin_compaction("a", 1, start_index=1)
    ledger.append_event("a", "m-2", {"role": "user", "content": "new message"})
    snapshot = ledger.commit_compaction(plan, {"role": "summary", "content": "summary"})
    assert snapshot.events[0].payload == system
    assert [event.event_id for event in snapshot.events] == ["system", f"compaction:{plan.token}", "m-2"]
    assert ledger.has_event("a", "m-1") and not ledger.has_event("b", "m-1")
    assert not ledger.has_event("a", "unknown-id")


def test_callback_receipt_precedes_actual_native_tool_result_without_fabrication(ledger):
    ledger.append_event("a", "call", native_call())
    ledger.prepare_tool_execution("a", "call-1", {"name": "reply"})
    ledger.claim_tool_execution("a", "call-1")
    before = ledger.snapshot("a")
    callback = {"content": [{"type": "text", "text": "sent"}], "details": {"native": {"items": []}}}
    receipt = ledger.record_tool_outcome("a", "call-1", callback)
    assert receipt.result == callback and receipt.result_event_id is None
    assert ledger.snapshot("a") == before  # Pi has not emitted its native message yet.
    assert ledger.record_tool_outcome("a", "call-1", callback) == receipt
    assert not ledger.claim_tool_execution("a", "call-1")
    with pytest.raises(UnsafeCompaction):
        ledger.begin_compaction("a", 1)
    actual_pi_message = {**native_result(), "timestamp": 1720307000123, "unknownPiField": "retain"}
    ledger.append_event("a", "actual-pi-result", actual_pi_message)
    assert ledger.snapshot("a").events[-1].payload == actual_pi_message
    assert len(ledger.begin_compaction("a", 2).prefix) == 2
    with pytest.raises(LedgerConflict, match="immutable"):
        ledger.record_tool_outcome("a", "call-1", {"different": True})


def test_mismatched_receipt_and_native_call_id_cannot_corrupt_pairing(ledger):
    ledger.prepare_tool_execution("a", "wrong-prefixed-call-1", {})
    ledger.claim_tool_execution("a", "wrong-prefixed-call-1")
    with pytest.raises(LedgerConflict, match="call id differs"):
        ledger.complete_tool_execution("a", "wrong-prefixed-call-1", "native-result", native_result("call-1"))
    assert ledger.snapshot("a").events == ()


def test_uncertain_receipt_fences_new_model_and_tool_callbacks_in_its_host_stream(ledger):
    ledger.assert_no_unresolved_executions("a")
    ledger.prepare_tool_execution("a", "call-1", {"name": "reply"})
    ledger.assert_no_unresolved_executions("a")  # No effect attempted yet.
    ledger.claim_tool_execution("a", "call-1")
    with pytest.raises(LedgerConflict, match="reconciliation"):
        ledger.assert_no_unresolved_executions("a")
    ledger.assert_no_unresolved_executions("other-real-session")
    ledger.mark_tool_execution_unknown("a", "call-1")
    # A provider-generated error reply is NOT evidence that the effect failed.
    ledger.append_event("a", "pi-error", {**native_result(), "isError": True})
    with pytest.raises(LedgerConflict, match="unknown"):
        ledger.assert_no_unresolved_executions("a")
    ledger.record_tool_outcome("a", "call-1", {"authoritativeReceipt": "sent"})
    ledger.assert_no_unresolved_executions("a")


def test_repeated_old_reference_read_cannot_supersede_newer_source_version(ledger):
    first = ledger.append_reference("a", "memory", "M1", "v1", "old")
    second = ledger.append_reference("a", "memory", "M1", "v2", "new")
    read = ledger.append_reference("a", "memory", "M1", "v1", "old", force=True)
    assert read.event.payload["supersedes"] is None
    assert read.event.payload["observation"] == "explicit_repeat"
    third = ledger.append_reference("a", "memory", "M1", "v3", "newest")
    assert third.event.payload["supersedes"] == second.event.event_id
    assert second.event.payload["supersedes"] == first.event.event_id


def test_pending_source_inbox_survives_luna_silence_cooldown_and_restart(ledger):
    payload = {"role": "user", "content": "arrived while cooldown", "native": {"full": ["original"]}}
    assert ledger.enqueue_input("host-session", "input:source:source-1", payload)
    pending = ledger.pending_inputs("host-session")
    assert len(pending) == 1 and pending[0].payload == payload
    assert ledger.snapshot("host-session").events == ()
    assert ledger.snapshot("host-session").revision == 0
    assert not ledger.has_event("host-session", "input:source:source-1")
    assert not ledger.enqueue_input("host-session", "input:source:source-1", payload)
    payload["native"]["full"].clear()
    restarted = ThreadLedger(ledger.db_path)
    assert restarted.pending_inputs("host-session") == pending
    assert restarted.pending_inputs("another-real-session") == ()
    with pytest.raises(LedgerConflict, match="changed content"):
        restarted.enqueue_input("host-session", "input:source:source-1", payload)
    snapshot = restarted.consume_input("host-session", "input:source:source-1", expected_revision=0)
    assert snapshot.events == pending and snapshot.revision == 1
    assert restarted.pending_inputs("host-session") == ()
    assert restarted.has_event("host-session", "input:source:source-1")
    assert restarted.consume_input("host-session", "input:source:source-1") == snapshot
    plan = restarted.begin_compaction("host-session", 1)
    compacted = restarted.commit_compaction(plan, {"role": "user", "kind": "summary", "content": "summary"})
    assert not restarted.enqueue_input("host-session", "input:source:source-1", pending[0].payload)
    assert restarted.consume_input("host-session", "input:source:source-1") == compacted


def test_pending_source_consumption_is_fifo_and_revision_atomic(ledger):
    ledger.enqueue_input("a", "source-1", {"role": "user", "content": "first"})
    ledger.enqueue_input("a", "source-2", {"role": "user", "content": "second"})
    before = ledger.pending_inputs("a")
    with pytest.raises(LedgerConflict, match="arrival order"):
        ledger.consume_input("a", "source-2")
    with pytest.raises(LedgerConflict, match="revision"):
        ledger.consume_input("a", "source-1", expected_revision=4)
    assert ledger.pending_inputs("a") == before
    first = ledger.consume_input("a", "source-1")
    assert first.events == before[:1]
    second = ledger.consume_input("a", "source-2")
    assert second.events == before and second.revision == 2
    with pytest.raises(KeyError):
        ledger.consume_input("a", "never-arrived")


def test_pending_input_waits_for_native_tool_result_and_known_effect_outcome(ledger):
    ledger.append_event("a", "call", native_call())
    ledger.prepare_tool_execution("a", "call-1", {"name": "reply"})
    ledger.claim_tool_execution("a", "call-1")
    ledger.enqueue_input("a", "new-source", {"role": "user", "content": "arrived during tool"})
    pending = ledger.pending_inputs("a")
    with pytest.raises(LedgerConflict, match="closed tool boundary"):
        ledger.consume_input("a", "new-source")
    ledger.append_event("a", "result", native_result())
    with pytest.raises(LedgerConflict, match="uncertain executions"):
        ledger.consume_input("a", "new-source")
    assert ledger.pending_inputs("a") == pending
    ledger.record_tool_outcome("a", "call-1", {"known": "sent"})
    snapshot = ledger.consume_input("a", "new-source")
    assert [event.event_id for event in snapshot.events] == ["call", "result", "new-source"]
    assert snapshot.events[-1] == pending[0]


def test_pending_intake_rollback_keeps_frozen_input_if_event_identity_conflicts(ledger):
    ledger.enqueue_input("a", "source-1", {"role": "user", "content": "frozen original"})
    pending = ledger.pending_inputs("a")
    before = ledger.append_event("a", "source-1", {"role": "user", "content": "conflicting event"})
    with pytest.raises(LedgerConflict, match="immutable event"):
        ledger.consume_input("a", "source-1")
    assert ledger.snapshot("a") == before
    assert ledger.pending_inputs("a") == pending


def test_concurrent_duplicate_source_deliveries_enqueue_once(ledger):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: ledger.enqueue_input("a", "source-1", {"role": "user", "content": "x"}), range(24))
        )
    assert sum(results) == 1
    assert len(ledger.pending_inputs("a")) == 1
    assert ledger.snapshot("a").revision == 0


def test_claim_atomically_fences_fresh_call_ids_when_another_effect_is_uncertain(ledger):
    for call_id in ("first", "second"):
        ledger.prepare_tool_execution("a", call_id, {"name": "reply"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(lambda call_id: ledger.claim_tool_execution("a", call_id), ("first", "second")))
    assert sum(claimed) == 1
    winner, waiting = ("first", "second") if claimed[0] else ("second", "first")
    ledger.mark_tool_execution_unknown("a", winner)
    assert not ledger.claim_tool_execution("a", waiting)
    ledger.prepare_tool_execution("child-host-scope", "child-call", {"name": "read"})
    assert ledger.claim_tool_execution("child-host-scope", "child-call")
    ledger.record_tool_outcome("a", winner, {"authoritative": "sent"})
    assert ledger.claim_tool_execution("a", waiting)


def test_unknown_partial_effect_observation_survives_restart_without_retry_permission(ledger):
    ledger.prepare_tool_execution("a", "send-call", {"name": "reply", "arguments": {"text": "two parts"}})
    assert ledger.claim_tool_execution("a", "send-call")
    observation = {
        "success": False,
        "error": "second part delivery ACK lost",
        "metadata": {"sentMessages": [{"messageId": "confirmed-first", "receipt": {"ack": True}}]},
        "content": [{"type": "text", "text": "first part sent; second outcome unknown"}],
    }
    receipt = ledger.mark_tool_execution_unknown("a", "send-call", observation=observation)
    assert receipt.status == "unknown" and receipt.result == observation
    assert receipt.result_event_id is None
    frozen_json = receipt.result_json
    observation["metadata"]["sentMessages"].clear()
    restarted = ThreadLedger(ledger.db_path)
    assert restarted.lookup_tool_execution("a", "send-call").result_json == frozen_json
    assert restarted.mark_tool_execution_unknown("a", "send-call") == receipt
    assert restarted.mark_tool_execution_unknown("a", "send-call", observation=None) == receipt
    assert not restarted.claim_tool_execution("a", "send-call")
    restarted.prepare_tool_execution("a", "fresh-call", {"name": "reply"})
    assert not restarted.claim_tool_execution("a", "fresh-call")
    with pytest.raises(LedgerConflict, match="reconciliation"):
        restarted.assert_no_unresolved_executions("a")
    assert restarted.snapshot("a").events == ()  # Partial evidence is not a fabricated terminal Pi message.
    additional = {"confirmedSentIds": ["confirmed-first", "confirmed-second"], "remainingOutcome": "uncertain"}
    updated = restarted.mark_tool_execution_unknown("a", "send-call", observation=additional)
    assert updated.result == additional and updated.status == "unknown"
    assert not restarted.claim_tool_execution("a", "send-call")


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_unknown_observation_cannot_reopen_a_known_terminal_outcome(ledger, status):
    ledger.prepare_tool_execution("a", "call", {"name": "reply"})
    ledger.claim_tool_execution("a", "call")
    known = ledger.record_tool_outcome("a", "call", {"definitive": status}, status=status)
    with pytest.raises(LedgerConflict, match="executing/unknown"):
        ledger.mark_tool_execution_unknown("a", "call", observation={"partial": "late evidence"})
    assert ledger.lookup_tool_execution("a", "call") == known
    assert not ledger.claim_tool_execution("a", "call")


def test_invalid_unknown_observation_does_not_change_execution_status(ledger):
    ledger.prepare_tool_execution("a", "call", {})
    ledger.claim_tool_execution("a", "call")
    with pytest.raises(TypeError, match="native JSON"):
        ledger.mark_tool_execution_unknown("a", "call", observation={"notJson": object()})
    assert ledger.lookup_tool_execution("a", "call").status == "executing"
