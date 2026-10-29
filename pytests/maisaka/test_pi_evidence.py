from types import SimpleNamespace

import asyncio
import json

import pytest

from src.maisaka.pi_runtime.evidence import DurableEvidenceStore, public_archive_payload
from src.maisaka.tool_result_store import ToolResultStore


def test_full_raw_survives_memory_eviction_and_restart_with_parent_child_scope(tmp_path):
    async def run():
        source = ToolResultStore()
        path = str(tmp_path / "evidence.sqlite")
        service = DurableEvidenceStore(path, source)
        raw = "preview\n" + "x" * 10000 + "\noriginal tail URL https://example.org/source"
        handle = source.save("parent", "web_fetch", raw)
        result = SimpleNamespace(metadata={"result_id": handle}, content="preview only")
        assert await service.capture("parent", result, child_namespace="child-1") == handle
        for _ in range(25):
            source.save("parent", "web_fetch", "new result")
        assert source.get("parent", handle) is None
        restarted = DurableEvidenceStore(path, ToolResultStore())
        parts, offset = [], 0
        while True:
            page = await restarted.read("parent", handle, offset, 999, child_namespace="child-1")
            parts.append(page["text"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        assert "".join(parts) == raw
        assert "original tail" in (await restarted.read("parent", handle, keyword="original tail"))["text"]
        with pytest.raises(PermissionError):
            await restarted.read("other-parent", handle)
        with pytest.raises(PermissionError):
            await restarted.read("parent", handle, child_namespace="child-2")
        with pytest.raises(RuntimeError):
            await restarted.capture("other-parent", result)
        foreign = source.save("foreign", "web_fetch", "foreign evidence")
        with pytest.raises(PermissionError):
            await service.capture("parent", SimpleNamespace(metadata={"result_id": foreign}), source_session_id="foreign")
    asyncio.run(run())


def test_archive_projection_never_exposes_replay_thinking_and_stays_scoped(tmp_path):
    from src.maisaka.pi_runtime.ledger import ThreadLedger

    async def run():
        ledger = await asyncio.to_thread(ThreadLedger, str(tmp_path / "ledger.sqlite"))
        payload = {"role": "assistant", "content": [{"type": "text", "text": "public answer"},
                    {"type": "thinking", "thinking": "PRIVATE"}], "native": {"items": "OPAQUE"}}
        await asyncio.to_thread(ledger.append_event, "parent", "event-1", payload)
        service = DurableEvidenceStore(str(tmp_path / "evidence.sqlite"))
        result = await service.read_archive(ledger, "parent", ["event-1"])
        assert "public answer" in result["text"]
        assert "PRIVATE" not in result["text"] and "OPAQUE" not in result["text"]
        with pytest.raises(KeyError):
            await service.read_archive(ledger, "other-parent", ["event-1"])
        assert "secret" not in json.dumps(public_archive_payload({"role": "system", "content": "secret"}))
    asyncio.run(run())


def test_archived_summary_preserves_next_hop_original_handles():
    from src.maisaka.pi_runtime.evidence import public_archive_payload
    result = public_archive_payload({
        "role": "user", "kind": "summary", "content": "earlier summary",
        "sourceArchive": {"streamId": "group", "eventIds": ["original-one", "original-two"]},
        "native": {"secret_replay": "never expose"},
    })
    assert result["source_archive"]["eventIds"] == ["original-one", "original-two"]
    assert "secret_replay" not in str(result)
