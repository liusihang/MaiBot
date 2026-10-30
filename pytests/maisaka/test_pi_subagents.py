"""No models or external tools are used by these delegation contract tests."""

from typing import Any, Dict

import asyncio
import json

import pytest

from src.maisaka.pi_runtime.subagents import MAX_CONCLUSION_CHARS, PiSubagentService


def tool(name: str) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def test_isolation_permissions_and_original_evidence(tmp_path):
    seen = []

    async def runner(child, parent, task, tools, event):
        seen.append((child, parent, task, tools))
        await event({"type": "message_end", "message": {"role": "toolResult", "content": "ORIGINAL"}})
        return {"conclusion": "supported", "sources": ["https://example.org/evidence"], "uncertainty": "date unknown"}

    async def run():
        service = PiSubagentService(str(tmp_path / "children.sqlite"), runner)
        result = await service.delegate(
            "real-group-1", "research", [tool("web_search"), tool("reply"), tool("unknown")]
        )
        assert [item["function"]["name"] for item in seen[0][3]] == ["web_search", "reply", "unknown"]
        assert seen[0][0] != seen[0][1] == "real-group-1"
        assert result["sources"] == ["https://example.org/evidence"]
        transcript = await service.read("real-group-1", result["handle"])
        assert "ORIGINAL" in transcript["text"]
        with pytest.raises(PermissionError):
            await service.read("other-group", result["handle"])
        for name in ("reply", "subagent", "remember_memory", "unknown"):
            service.assert_tool_allowed(name, [name])
        with pytest.raises(PermissionError):
            service.assert_tool_allowed("web_fetch", ["web_search"])
        service.assert_tool_allowed("web_search", ["web_search"])

    asyncio.run(run())


def test_no_eviction_restart_pagination_and_bounded_parent_output(tmp_path):
    async def runner(child, parent, task, tools, event):
        await event(
            {
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": "original-evidence-" + "x" * 15000}],
                    "native": {"replay": "opaque-private"},
                },
            }
        )
        await event(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "PRIVATE_THINKING"},
                    ],
                },
            }
        )
        return {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "a" * 9000}]}]}

    async def run():
        path = str(tmp_path / "children.sqlite")
        service = PiSubagentService(path, runner)
        first = await service.delegate("parent", "task", [])
        assert len(first["conclusion"]) == MAX_CONCLUSION_CHARS
        assert first["conclusion_truncated"] is True
        assert first["sources"] == []
        assert "No separately structured" in first["uncertainty"]
        for _ in range(22):
            await service.delegate("parent", "later task", [])
        restarted = PiSubagentService(path, runner)
        pages, offset = [], 0
        while True:
            page = await restarted.read("parent", first["handle"], offset, 999)
            assert len(page["text"]) <= 999
            pages.append(page["text"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        original = json.loads("".join(pages))
        assert original["events"][0]["text"] == "original-evidence-" + "x" * 15000
        assert len(original["result"]["conclusion"]) == 9000
        assert "opaque-private" not in "".join(pages)
        assert "PRIVATE_THINKING" not in "".join(pages)
        # Provider packets remain only in Node SessionManager, not the Python business cache.
        with restarted._connect() as connection:
            stored = connection.execute(
                "SELECT payload FROM pi_child_events WHERE handle=?", (first["handle"],)
            ).fetchall()
            assert "opaque-private" not in stored[0][0]
            assert "PRIVATE_THINKING" not in str(stored)
        with pytest.raises(ValueError):
            await restarted.read("parent", first["handle"], -1)
        with pytest.raises(ValueError):
            await restarted.read("parent", first["handle"], limit=12001)

    asyncio.run(run())


def test_failed_and_cancelled_child_evidence_survives(tmp_path):
    captured = []

    async def runner(child, parent, task, tools, event):
        await event({"progress": task})
        if task == "cancel":
            raise asyncio.CancelledError()
        raise RuntimeError("failed")

    async def run():
        service = PiSubagentService(str(tmp_path / "children.sqlite"), runner)
        create = service._create

        def capture(handle, *args):
            captured.append(handle)
            create(handle, *args)

        service._create = capture
        failure = await service.delegate("parent", "fail", [])
        assert failure["status"] == "failed" and failure["error"] == "RuntimeError"
        with pytest.raises(asyncio.CancelledError):
            await service.delegate("parent", "cancel", [])
        for handle, expected in zip(captured, ["failed", "cancelled"], strict=True):
            record = json.loads((await service.read("parent", handle))["text"])
            assert record["status"] == expected
            assert record["events"] == []

    asyncio.run(run())


@pytest.mark.asyncio
async def test_background_returns_while_work_runs_and_wait_joins_without_polling(tmp_path):
    release = asyncio.Event()
    notifications = []

    async def runner(child, parent, task, tools, event):
        await release.wait()
        await event({"type": "message_end", "message": {"role": "toolResult", "content": "sent image receipt"}})
        return {"conclusion": "image sent", "sources": ["https://example.org/source"]}

    async def notify(result):
        notifications.append(result)

    service = PiSubagentService(tmp_path / "children.sqlite", runner, notify)
    result = await service.delegate("group", "edit and send", [tool("gemini_generate_image")], wait=False)
    assert result["status"] == "running"
    with pytest.raises(PermissionError):
        await service.read("other-group", result["handle"], wait=True)
    join = asyncio.create_task(service.read("group", result["handle"], wait=True))
    await asyncio.sleep(0)
    assert not join.done()
    release.set()
    completed = await join
    assert completed["status"] == "completed" and completed["conclusion"] == "image sent"
    assert notifications[0]["handle"] == result["handle"]
    await service.recover("group")
    assert len(notifications) == 1
    await service.close()


@pytest.mark.asyncio
async def test_restart_recovers_unnotified_completion_and_marks_interrupted_actions(tmp_path):
    path = tmp_path / "children.sqlite"

    async def runner(*args):
        return {"conclusion": "done"}

    first = PiSubagentService(path, runner)
    first._create("finished", "group", "child-1", "draw", ["reply"], True)
    first._finish("finished", "completed", {"conclusion": "sent once"})
    first._create("lost", "group", "child-2", "edit", ["gemini_generate_image"], True)
    notifications = []

    async def notify(result):
        notifications.append(result)

    restarted = PiSubagentService(path, runner, notify)
    await restarted.recover("group")
    assert {r["handle"]: r["status"] for r in notifications} == {"finished": "completed", "lost": "interrupted"}
    assert restarted._tasks == {}
    await restarted.recover("group")
    assert len(notifications) == 2
