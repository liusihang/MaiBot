"""Pi children must feed the existing sidebar without publishing parent rounds."""

from unittest.mock import AsyncMock

import asyncio

import pytest

from src.maisaka.monitor import events
from src.maisaka.pi_runtime.subagents import PiSubagentService


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_child_lifecycle_and_public_tools_reach_sidebar(monkeypatch, tmp_path, fail):
    broadcast = AsyncMock()
    monkeypatch.setattr(events, "_broadcast", broadcast)

    async def child(child_id, parent, task, tools, on_event):
        await on_event({"type": "message_end", "message": {
            "role": "assistant", "content": [
                {"type": "thinking", "thinking": "PRIVATE_REASONING", "signature": "PRIVATE_SIGNATURE"},
                {"type": "toolCall", "id": "search-call", "name": "web_fetch", "arguments": {"url": "https://example.org"}},
            ],
        }})
        await on_event({"type": "tool_execution_start", "toolCallId": "search-call"})
        await on_event({"type": "message_end", "message": {
            "role": "toolResult", "toolCallId": "search-call", "toolName": "web_fetch",
            "content": [{"type": "text", "text": "PUBLIC_SOURCE"}], "isError": fail,
        }})
        if fail:
            raise RuntimeError("child failed")
        return {"conclusion": "PUBLIC_FINDINGS"}

    service = PiSubagentService(str(tmp_path / "children.sqlite"), child)
    result = await service.delegate("real-parent", "read article", [])
    observed = [(call.args[0], call.args[1]) for call in broadcast.await_args_list]
    assert [name for name, _ in observed] == [
        "subagent.started", "subagent.round", "subagent.tool_result", "subagent.finished",
    ]
    assert {data["session_id"] for _, data in observed} == {"real-parent"}
    assert {data["task_id"] for _, data in observed} == {result["handle"]}
    assert observed[1][1]["round_index"] == observed[2][1]["round_index"] == 1
    assert observed[1][1]["tool_calls"][0]["id"] == observed[2][1]["call_id"] == "search-call"
    assert observed[2][1]["success"] is (not fail)
    assert observed[-1][1]["status"] == ("failed" if fail else "completed")
    assert observed[-1][1]["tool_usage"] == {"web_fetch": 1}
    assert "PRIVATE_" not in str(observed)


@pytest.mark.asyncio
async def test_monitor_failure_cannot_fail_a_child(monkeypatch, tmp_path):
    monkeypatch.setattr(events, "_broadcast", AsyncMock(side_effect=OSError("monitor unavailable")))

    async def child(*args):
        return {"conclusion": "original findings"}

    service = PiSubagentService(str(tmp_path / "children.sqlite"), child)
    result = await service.delegate("real-parent", "research", [])
    assert result["status"] == "completed" and result["conclusion"] == "original findings"


@pytest.mark.asyncio
async def test_cancelled_child_finishes_its_sidebar_task(monkeypatch, tmp_path):
    broadcast, ready = AsyncMock(), asyncio.Event()
    monkeypatch.setattr(events, "_broadcast", broadcast)

    async def child(*args):
        ready.set()
        await asyncio.Event().wait()

    service = PiSubagentService(str(tmp_path / "children.sqlite"), child)
    result = await service.delegate("real-parent", "research", [], wait=False)
    await ready.wait()
    await service.close()
    terminal = [call.args[1] for call in broadcast.await_args_list if call.args[0] == "subagent.finished"]
    assert len(terminal) == 1 and terminal[0]["task_id"] == result["handle"]
    assert terminal[0]["status"] == "stopped"
