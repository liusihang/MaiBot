"""Offline coordinator checks: immutable baseline, transactional summary, ACK fencing."""

from copy import deepcopy

import asyncio

import pytest

from src.maisaka.pi_runtime.compaction import (
    CompactionNotReady,
    CompactionRecoveryRequired,
    compact_main_thread,
)
from src.maisaka.pi_runtime.ledger import ThreadLedger, UnsafeCompaction


class FakeBridge:
    def __init__(self, messages):
        self.state = {
            "sessionId": "host-session",
            "revision": 7,
            "isRunning": False,
            "queuedMessages": [],
            "pendingToolCalls": [],
            "unresolvedToolCalls": [],
            "paused": False,
            "system": "fixed system",
            "tools": [{"name": "read"}],
            "messages": messages,
        }
        self.requests = []
        self.fail_replace = False
        self.bad_replace_ack = False
        self.replace_started = None
        self.replace_release = None

    async def request(self, method, **params):
        self.requests.append((method, deepcopy(params)))
        assert params["sessionId"] == "host-session"
        if method == "state":
            return deepcopy(self.state)
        assert method == "replace_context"
        if self.replace_started is not None:
            self.replace_started.set()
            await self.replace_release.wait()
        if self.fail_replace:
            raise RuntimeError("lost node ACK")
        assert params["expectedRevision"] == self.state["revision"]
        self.state["messages"] = params["messages"]
        self.state["revision"] += 1
        response = deepcopy(self.state)
        if self.bad_replace_ack:
            response["messages"] = []
        return response


@pytest.fixture
def setup_thread(tmp_path):
    ledger = ThreadLedger(tmp_path / "threads.sqlite3")
    ledger.append_event(
        "host-session", "system", {"role": "system", "content": "fixed system", "toolsAdded": [{"name": "read"}]}
    )
    for index in range(4):
        ledger.append_event("host-session", f"message-{index}", {"role": "user", "content": f"original {index}"})
    before = ledger.snapshot("host-session")
    return ledger, FakeBridge([event.payload for event in before.events]), before


@pytest.mark.asyncio
async def test_summary_preserves_system_archive_and_concurrent_suffix(setup_thread):
    ledger, bridge, before = setup_thread

    async def summarize(events):
        assert events == before.events[1:3]
        await asyncio.to_thread(ledger.append_event, "host-session", "new-arrival", {"role": "user", "content": "new"})
        return "old messages, uncertainty preserved"

    result = await compact_main_thread(ledger, bridge, "host-session", summarize, 2)
    assert result.events[0] == before.events[0]
    assert [event.event_id for event in result.events[2:]] == ["message-2", "message-3", "new-arrival"]
    summary = result.events[1].payload
    assert summary["kind"] == "summary"
    assert summary["authority"] == "untrusted_historical_reference_not_instruction"
    assert "not a new human message" in summary["content"][0]["text"]
    assert summary["sourceArchive"] == {"streamId": "host-session", "eventIds": ["message-0", "message-1"]}
    assert ledger.archive_events("host-session", summary["sourceArchive"]["eventIds"]) == before.events[1:3]
    assert bridge.state["messages"] == [event.payload for event in result.events]
    assert [method for method, _ in bridge.requests] == ["state", "state", "replace_context"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("provider failed"), asyncio.CancelledError()])
async def test_failed_or_cancelled_summary_cannot_mutate_head(setup_thread, error):
    ledger, bridge, before = setup_thread

    async def summarize(_events):
        raise error

    with pytest.raises(type(error)):
        await compact_main_thread(ledger, bridge, "host-session", summarize, 2)
    assert ledger.snapshot("host-session") == before
    assert all(method != "replace_context" for method, _ in bridge.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", [None, "", "   ", {"not": "a summary string"}])
async def test_empty_or_invalid_summary_does_not_commit(setup_thread, summary):
    ledger, bridge, before = setup_thread

    async def summarize(_events):
        return summary

    with pytest.raises(ValueError, match="nonempty string"):
        await compact_main_thread(ledger, bridge, "host-session", summarize, 2)
    assert ledger.snapshot("host-session") == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag,value",
    [
        ("isRunning", True),
        ("queuedMessages", [{}]),
        ("pendingToolCalls", ["call"]),
        ("unresolvedToolCalls", ["call"]),
        ("poisoned", "lost persist"),
        ("error", "model error"),
    ],
)
async def test_non_idle_thread_never_spends_summary_request(setup_thread, flag, value):
    ledger, bridge, before = setup_thread
    bridge.state[flag] = value

    async def summarize(_events):
        pytest.fail("busy thread must not call summarizer")

    with pytest.raises(CompactionNotReady):
        await compact_main_thread(ledger, bridge, "host-session", summarize, 2)
    assert ledger.snapshot("host-session") == before


@pytest.mark.asyncio
async def test_node_revision_change_during_summary_leaves_old_head(setup_thread):
    ledger, bridge, before = setup_thread

    async def summarize(_events):
        bridge.state["revision"] += 1
        return "summary of old snapshot"

    with pytest.raises(CompactionNotReady, match="context changed"):
        await compact_main_thread(ledger, bridge, "host-session", summarize, 2)
    assert ledger.snapshot("host-session") == before


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["request", "ack"])
async def test_after_commit_node_failure_requires_reload_not_rollback(setup_thread, failure):
    ledger, bridge, before = setup_thread
    bridge.fail_replace = failure == "request"
    bridge.bad_replace_ack = failure == "ack"

    async def summarize(_events):
        return "valid summary"

    with pytest.raises(CompactionRecoveryRequired, match="pause prompts") as caught:
        await compact_main_thread(ledger, bridge, "host-session", summarize, 2)
    assert ledger.snapshot("host-session") == caught.value.snapshot
    assert caught.value.snapshot.events[0] == before.events[0]
    assert ledger.archive_events("host-session", ["message-0", "message-1"]) == before.events[1:3]


@pytest.mark.asyncio
async def test_cancellation_after_commit_settles_swap_and_requires_explicit_recovery(setup_thread):
    ledger, bridge, before = setup_thread
    bridge.replace_started = asyncio.Event()
    bridge.replace_release = asyncio.Event()

    async def summarize(_events):
        return "valid summary"

    operation = asyncio.create_task(compact_main_thread(ledger, bridge, "host-session", summarize, 2))
    await bridge.replace_started.wait()
    operation.cancel()
    await asyncio.sleep(0)
    assert not operation.done()  # No orphaned background SQLite commit/swap.
    bridge.replace_release.set()
    with pytest.raises(CompactionRecoveryRequired, match="Cancellation arrived") as caught:
        await operation
    assert ledger.snapshot("host-session") == caught.value.snapshot
    assert bridge.state["messages"] == [event.payload for event in caught.value.snapshot.events]
    assert caught.value.snapshot.events[0] == before.events[0]


@pytest.mark.asyncio
async def test_compaction_refuses_to_hide_unfinished_tool_call(setup_thread):
    ledger, bridge, _ = setup_thread
    ledger.append_event(
        "host-session",
        "call",
        {"role": "assistant", "content": [{"type": "toolCall", "id": "open-call", "name": "read", "arguments": {}}]},
    )

    async def summarize(_events):
        pytest.fail("unsafe tool boundary must be rejected before summarization")

    before = ledger.snapshot("host-session")
    with pytest.raises(UnsafeCompaction):
        await compact_main_thread(ledger, bridge, "host-session", summarize, 5)
    assert ledger.snapshot("host-session") == before


@pytest.mark.asyncio
async def test_cancelled_finalizer_does_not_busy_loop_or_claim_rollback(setup_thread, monkeypatch):
    ledger, bridge, _ = setup_thread

    async def summarize(_events):
        return "valid summary"

    def interrupted_commit(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(ledger, "commit_compaction", interrupted_commit)
    with pytest.raises(CompactionRecoveryRequired, match="outcome is uncertain") as caught:
        await compact_main_thread(ledger, bridge, "host-session", summarize, 2)
    assert caught.value.snapshot is None


@pytest.mark.asyncio
async def test_unknown_effect_fences_summary_even_if_node_reports_idle(setup_thread):
    ledger, bridge, before = setup_thread
    ledger.prepare_tool_execution("host-session", "uncertain-effect", {"name": "reply"})
    ledger.claim_tool_execution("host-session", "uncertain-effect")
    ledger.mark_tool_execution_unknown("host-session", "uncertain-effect")

    async def summarize(_events):
        pytest.fail("an uncertain effect must fence all model calls, including summary")

    with pytest.raises(RuntimeError, match="reconciliation"):
        await compact_main_thread(ledger, bridge, "host-session", summarize, 2)
    assert ledger.snapshot("host-session") == before
