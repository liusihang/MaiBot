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
        assert [item["function"]["name"] for item in seen[0][3]] == ["web_search"]
        assert seen[0][0] != seen[0][1] == "real-group-1"
        assert result["sources"] == ["https://example.org/evidence"]
        transcript = await service.read("real-group-1", result["handle"])
        assert "ORIGINAL" in transcript["text"]
        with pytest.raises(PermissionError):
            await service.read("other-group", result["handle"])
        for name in ("reply", "subagent", "remember_memory", "unknown"):
            with pytest.raises(PermissionError):
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
        with pytest.raises(RuntimeError):
            await service.delegate("parent", "fail", [])
        with pytest.raises(asyncio.CancelledError):
            await service.delegate("parent", "cancel", [])
        for handle, expected in zip(captured, ["failed", "cancelled"], strict=True):
            record = json.loads((await service.read("parent", handle))["text"])
            assert record["status"] == expected
            assert record["events"] == []

    asyncio.run(run())
