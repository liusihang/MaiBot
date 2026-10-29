"""Compaction is a transaction: failures/cancellation never commit planned loss."""

from datetime import datetime
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import asyncio

import pytest

from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.llm_models.payload_content.context_item import ContextItemMeta, ContextToolCall, FunctionCallItem
from src.maisaka import reasoning_engine as engine_module
from src.maisaka.context.compaction import ContextSummaryMessage
from src.maisaka.context.messages import (
    ModelOutputContextMessage,
    ReferenceMessage,
    SessionBackedMessage,
    ToolResultMessage,
)
from src.maisaka.reasoning_engine import MaisakaReasoningEngine


def chat_message(index):
    source = SessionMessage(f"m-{index}", datetime(2026, 1, 1), "offline")
    source.session_id = "offline-session"
    source.message_info = MessageInfo(UserInfo("A", "A"))
    source.processed_plain_text = f"original text {index}"
    source.raw_message = MessageSequence([TextComponent(source.processed_plain_text)])
    return SessionBackedMessage.from_session_message(
        source,
        raw_message=source.raw_message,
        visible_text=source.processed_plain_text,
    )


def setup_engine(monkeypatch, *, count=8):
    ref = ReferenceMessage("one-shot state", datetime(2026, 1, 1), remaining_uses_value=2)
    history = [chat_message(index) for index in range(count)] + [ref]
    snapshot = object()
    runtime = NS(
        session_id="offline-session",
        log_prefix="[offline]",
        _chat_history=history,
        _max_context_size=5,
        _get_pending_wait_tool_call_ids=lambda: set(),
        _chat_loop_service=NS(planner_request_snapshot=snapshot, context_budget_exceeded=True),
        _monitor_run_id="offline-run",
        _log_history_trimmed=Mock(),
        _trigger_trimmed_history_learning=AsyncMock(),
    )
    monkeypatch.setattr(engine_module, "emit_flow_step", AsyncMock())
    return MaisakaReasoningEngine(runtime), runtime, history, ref, snapshot


def cycle():
    return NS(cycle_id=1, time_records={})


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("offline failure"), None])
async def test_summary_failure_keeps_originals_and_skips_identical_failed_batch(monkeypatch, failure):
    engine, runtime, original, ref, snapshot = setup_engine(monkeypatch)
    summary = AsyncMock(side_effect=failure) if failure is not None else AsyncMock(return_value=None)
    monkeypatch.setattr(engine_module, "build_context_summary", summary)
    saved = list(original)
    await engine._post_process_chat_history_after_cycle(cycle())
    assert runtime._chat_history is original
    assert runtime._chat_history == saved
    assert ref.remaining_uses_value == 2
    assert runtime._chat_loop_service.planner_request_snapshot is snapshot
    assert runtime._chat_loop_service.context_budget_exceeded is True
    runtime._trigger_trimmed_history_learning.assert_not_called()
    await engine._post_process_chat_history_after_cycle(cycle())
    # An updated planner request snapshot or unrelated tail reference is not a changed old batch.
    runtime._chat_loop_service.planner_request_snapshot = object()
    runtime._chat_history.append(ReferenceMessage("new tail only", datetime.now(), remaining_uses_value=None))
    await engine._post_process_chat_history_after_cycle(cycle())
    assert summary.await_count == 1
    assert ref.remaining_uses_value == 2
    # In-place changed source text is a genuinely different compaction input and may retry.
    original[0].visible_text = "corrected source content"
    await engine._post_process_chat_history_after_cycle(cycle())
    assert summary.await_count == 2
    assert original[0] in runtime._chat_history


@pytest.mark.asyncio
async def test_success_commits_summary_and_reference_lifecycle_once(monkeypatch):
    engine, runtime, original, ref, _ = setup_engine(monkeypatch)
    summary = ContextSummaryMessage("verified structured summary output", datetime.now())
    build = AsyncMock(return_value=summary)
    monkeypatch.setattr(engine_module, "build_context_summary", build)
    engine._failed_compaction_batch = "different-old-failure"
    await engine._post_process_chat_history_after_cycle(cycle())
    assert runtime._chat_history[0] is summary
    assert runtime._chat_history[1:] == original[3:]
    assert ref.remaining_uses_value == 1
    assert build.call_args.args[0] == original[:3]
    assert runtime._chat_loop_service.planner_request_snapshot is None
    assert runtime._chat_loop_service.context_budget_exceeded is False
    assert engine._failed_compaction_batch is None
    runtime._trigger_trimmed_history_learning.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["append", "clear", "reference_lifecycle", "content"])
@pytest.mark.parametrize("fail", [False, True])
async def test_concurrent_update_is_not_overwritten_or_rolled_back(monkeypatch, change, fail):
    engine, runtime, original, ref, snapshot = setup_engine(monkeypatch)
    arrival = chat_message("new")

    async def summarize(*args, **kwargs):
        if change == "append":
            runtime._chat_history.append(arrival)
        elif change == "clear":
            runtime._chat_history = []
        elif change == "content":
            original[0].visible_text = "source text changed during summary request"
        else:
            ref.consume_once()
            ref.consume_once()
        if fail:
            raise RuntimeError("offline failure after concurrent update")
        return ContextSummaryMessage("old snapshot summary", datetime.now())

    monkeypatch.setattr(engine_module, "build_context_summary", summarize)
    await engine._post_process_chat_history_after_cycle(cycle())
    if change == "append":
        assert runtime._chat_history is original and runtime._chat_history[-1] is arrival
    elif change == "clear":
        assert runtime._chat_history == []
    else:
        assert runtime._chat_history is original
    assert ref.remaining_uses_value == (0 if change == "reference_lifecycle" else 2)
    assert runtime._chat_loop_service.planner_request_snapshot is snapshot
    runtime._trigger_trimmed_history_learning.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_summary_does_not_consume_reference_or_trim(monkeypatch):
    engine, runtime, original, ref, _ = setup_engine(monkeypatch)
    monkeypatch.setattr(engine_module, "build_context_summary", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await engine._post_process_chat_history_after_cycle(cycle())
    assert runtime._chat_history is original
    assert ref.remaining_uses_value == 2
    assert engine._failed_compaction_batch is None


@pytest.mark.asyncio
async def test_noncompacting_cycle_consumes_reference_even_when_no_structure_changes(monkeypatch):
    engine, runtime, original, ref, _ = setup_engine(monkeypatch, count=2)
    summary = AsyncMock()
    monkeypatch.setattr(engine_module, "build_context_summary", summary)
    await engine._post_process_chat_history_after_cycle(cycle())
    assert ref.remaining_uses_value == 1 and ref in runtime._chat_history
    await engine._post_process_chat_history_after_cycle(cycle())
    assert ref.remaining_uses_value == 0 and ref not in runtime._chat_history
    assert runtime._chat_history == original[:2]
    summary.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("force_archive", [False, True])
async def test_disabled_compaction_cannot_commit_count_trim_or_budget_archive(monkeypatch, force_archive):
    engine, runtime, original, ref, snapshot = setup_engine(monkeypatch, count=2 if force_archive else 8)
    runtime._chat_loop_service.context_budget_exceeded = force_archive
    if force_archive:
        old_call = ModelOutputContextMessage(
            FunctionCallItem(
                meta=ContextItemMeta.create(logical_turn_id="completed-old-turn"),
                tool_call=ContextToolCall.create(call_id="old-call", func_name="query_memory", args={"query": "old"}),
            )
        )
        old_result = ToolResultMessage(
            "old source evidence", datetime.now(), "old-call", "completed-old-turn", "query_memory"
        )
        original[1:1] = [old_call, old_result]
        engine._active_logical_turn_id = "current-turn"
    summary = AsyncMock(side_effect=AssertionError("silent/ignored turn must not spend a model call"))
    monkeypatch.setattr(engine_module, "build_context_summary", summary)
    saved = list(original)
    await engine._post_process_chat_history_after_cycle(cycle(), enable_compaction=False)
    assert runtime._chat_history is original
    assert runtime._chat_history == saved
    assert ref.remaining_uses_value == 2
    assert runtime._chat_loop_service.planner_request_snapshot is snapshot
    assert runtime._chat_loop_service.context_budget_exceeded is force_archive
    runtime._trigger_trimmed_history_learning.assert_not_called()
    summary.assert_not_called()
