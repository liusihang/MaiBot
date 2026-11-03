from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.maisaka.pi_runtime import observer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("recovering", "正文已生成但未发送"),
        ("undelivered", "正文未发送"),
        ("sent", "发送工具成功回执"),
    ],
)
async def test_monitor_distinguishes_generated_text_from_actual_delivery(monkeypatch, status, expected):
    emit = AsyncMock()
    monkeypatch.setattr(observer, "emit_planner_snapshot", emit)
    runtime = SimpleNamespace(session_id="group", _cycle_counter=1, _monitor_run_id="run", _agent_state="running")
    monitor = observer.PiObserver(runtime)
    monitor.event = {
        "message": {"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": "draft"}]}
    }
    await monitor.handle({"type": "turn_end", "deliveryStatus": status})
    final = emit.call_args.kwargs
    assert final["planner_content"] == "draft"
    assert expected in final["end_detail"]
    assert final["tools"] == []


@pytest.mark.asyncio
async def test_monitor_marks_a_finished_reply_as_sent_and_complete(monkeypatch):
    emit = AsyncMock()
    monkeypatch.setattr(observer, "emit_planner_snapshot", emit)
    runtime = SimpleNamespace(session_id="group", _cycle_counter=1, _monitor_run_id="run", _agent_state="running")
    monitor = observer.PiObserver(runtime)
    monitor.event = {"message": {"role": "assistant", "stopReason": "toolUse", "content": []}}
    await monitor.handle({"type": "turn_end", "deliveryStatus": "sent", "finishedAfterReply": True})
    assert emit.call_args.kwargs["end_reason"] == "reply_complete"
    assert emit.call_args.kwargs["end_detail"] == "本轮回复发送完成，已结束处理"
