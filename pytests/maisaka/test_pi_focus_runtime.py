"""Focus schedules business batches; isolated Pi runners own all inference."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncio

import pytest

from src.maisaka.context.compaction import ContextSummaryMessage
from src.maisaka.context.messages import ReferenceMessage
from src.maisaka.focus.context import FocusContextEntry
from src.maisaka.focus.group_runtime import FocusGroupRuntime


def notice(text):
    return ReferenceMessage(content=text, timestamp=datetime.now(), remaining_uses_value=None)


@pytest.fixture
def focused():
    group = FocusGroupRuntime("isolated-focus")
    group._hydrating = True
    group.checkpoint = AsyncMock()
    group._service.chat_loop_step = AsyncMock(side_effect=AssertionError("Shared planner must not run"))
    group._compact = AsyncMock(side_effect=AssertionError("Python shared compaction must not run"))
    group._build_references = AsyncMock(side_effect=AssertionError("Cross-chat references must not be built"))
    group.is_member_valid = lambda runtime: runtime._running and runtime._focus_group is group
    members = {}
    for sid in ("A", "B"):
        runner = SimpleNamespace(record_pending=AsyncMock(), run=AsyncMock())
        sources = {}
        member = SimpleNamespace(
            session_id=sid,
            session_name="chat-" + sid,
            _running=True,
            _focus_group=group,
            _chat_history=[],
            _agent_state="stop",
            _STATE_RUNNING="running",
            _STATE_WAIT="wait",
            _STATE_STOP="stop",
            _update_stage_status=Mock(),
            _collect_pending_messages=Mock(),
            _clear_forced_turn_state=Mock(),
            _reasoning_engine=SimpleNamespace(_get_pi_runner=lambda runner=runner: runner),
            find_source_message_by_id=lambda key, sources=sources: sources.get(key),
            sources=sources,
            runner=runner,
        )
        members[sid] = member
        group.members[sid] = member
    return group, members


def add_input(group, member, number, *, pending=False):
    source = SimpleNamespace(session_id=member.session_id, message_id=f"{member.session_id}-{number}")
    member.sources[source.message_id] = source
    entry = FocusContextEntry(
        [], {member.session_id}, number, inputs={member.session_id: [(number, source.message_id)]}
    )
    if pending:
        group._pending.setdefault(member.session_id, []).append(entry)
    else:
        group.entries.append(entry)
    group._sequence = max(group._sequence, number)
    return source


@pytest.mark.asyncio
async def test_focus_routes_each_snapshot_only_to_its_group_runner_and_never_mixed_legacy_context(focused):
    group, members = focused
    a = add_input(group, members["A"], 1)
    b = add_input(group, members["B"], 2)
    group.entries.extend(
        [
            FocusContextEntry([notice("private A notice")], {"A"}, 3),
            FocusContextEntry([notice("old mixed private summary")], {"A", "B"}, 4),
            FocusContextEntry([ContextSummaryMessage(content="legacy summary A", timestamp=datetime.now())], {"A"}, 5),
            FocusContextEntry([SimpleNamespace(role="assistant")], {"A"}, 6),
        ]
    )
    await group._run_turn()
    members["A"].runner.run.assert_awaited_once_with([a])
    members["B"].runner.run.assert_awaited_once_with([b])
    members["A"].runner.record_pending.assert_awaited_once_with(a)
    members["B"].runner.record_pending.assert_awaited_once_with(b)
    assert [message.content for message in members["A"]._chat_history] == ["private A notice"]
    assert members["B"]._chat_history == []
    group._service.chat_loop_step.assert_not_awaited()
    group._compact.assert_not_awaited()
    group._build_references.assert_not_awaited()
    assert group._considered == {"A": 6, "B": 4}


@pytest.mark.asyncio
async def test_focus_arrival_during_another_chat_keeps_its_own_next_ready_watermark(focused):
    group, members = focused
    a = add_input(group, members["A"], 1)
    b = add_input(group, members["B"], 2)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def run_a(sources):
        assert sources == [a]
        entered.set()
        await release.wait()

    members["A"].runner.run.side_effect = run_a
    running = asyncio.create_task(group._run_turn())
    await entered.wait()
    newer = add_input(group, members["B"], 3, pending=True)
    release.set()
    await running
    members["B"].runner.run.assert_awaited_once_with([b])
    assert group._considered["B"] == 2
    assert group._pending["B"][0].inputs == {"B": [(3, newer.message_id)]}
    group._ready.add("B")
    await group._run_turn()
    assert members["B"].runner.run.await_args.args == ([newer],)
    assert group._considered["B"] == 3


@pytest.mark.asyncio
async def test_focus_failure_acknowledges_only_prior_successful_chat_and_retains_failed_sources(focused):
    group, members = focused
    add_input(group, members["A"], 1)
    b = add_input(group, members["B"], 2)
    members["B"].runner.run.side_effect = RuntimeError("unknown business effect")
    with pytest.raises(RuntimeError, match="unknown business"):
        await group._run_turn()
    assert group._considered == {"A": 1}
    assert group.entries[1].inputs == {"B": [(2, b.message_id)]}
    assert await group.unprocessed_messages("B", through_sequence=2) == [b]


@pytest.mark.asyncio
async def test_focus_cancel_or_generation_change_never_marks_a_batch_processed(focused):
    group, members = focused
    source = add_input(group, members["A"], 1)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def run_a(_sources):
        entered.set()
        await release.wait()

    members["A"].runner.run.side_effect = run_a
    task = asyncio.create_task(group._run_turn())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not group._considered
    assert group.entries[0].inputs == {"A": [(1, source.message_id)]}
    release.set()

    async def invalidate(_sources):
        group._generation += 1

    members["A"].runner.run.side_effect = invalidate
    await group._run_turn()
    assert not group._considered


@pytest.mark.asyncio
async def test_focus_group_wait_stops_later_chat_and_resumes_after_scoped_wait_notice(focused):
    group, members = focused
    add_input(group, members["A"], 1)
    b = add_input(group, members["B"], 2)
    members["A"].runner.run.side_effect = lambda _sources: group.execute_control("wait", {"seconds": 60}, "A")
    await group._run_turn()
    assert group._wait_session_id == "A"
    assert group._considered == {"A": 1}
    members["B"].runner.run.assert_not_awaited()
    members["A"].runner.run.side_effect = None
    group._finish_wait("a mention woke the scheduler")
    await group._run_turn()
    members["B"].runner.run.assert_awaited_once_with([b])
    assert any("本聊天" in message.content for message in members["A"]._chat_history)
    assert not members["B"]._chat_history


@pytest.mark.asyncio
async def test_focus_retry_keeps_singleton_notice_identity_and_rejects_wrong_source_owner(focused):
    group, members = focused
    group.entries.append(FocusContextEntry([notice("stable event")], {"A"}, 1))
    members["A"].runner.run.side_effect = RuntimeError("temporary failed run")
    with pytest.raises(RuntimeError):
        await group._run_turn()
    members["A"].runner.run.side_effect = None
    await group._run_turn()
    assert len(members["A"]._chat_history) == 1
    wrong = add_input(group, members["B"], 2)
    wrong.session_id = "A"
    with pytest.raises(PermissionError, match="目标聊天"):
        await group._run_turn()
