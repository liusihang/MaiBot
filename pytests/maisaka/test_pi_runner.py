from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import shutil

import pytest

from src.common.data_models.message_component_data_model import MessageSequence
from src.core.tooling import ToolExecutionResult, ToolSpec
from src.llm_models.payload_content.context_item import (
    AssistantMessageItem, ContextItemMeta, ContextTextPart, ContextToolCall, FunctionCallItem,
)
from src.maisaka.context.messages import ContentContextMessage
from src.maisaka.pi_runtime.codec import encode_items
from src.maisaka.pi_runtime.provider import context_from_pi
from src.maisaka.pi_runtime.runner import PiGroupRunner

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Pi needs Node >=22.19")


def history(text, identifier):
    return ContentContextMessage(raw_message=MessageSequence([]), visible_text=text, timestamp=datetime.now(), context_item_id=identifier)


def engine():
    invoked = []
    specs = [ToolSpec(name=name, description=name, parameters_schema={"type": "object", "properties": {}, "additionalProperties": True}, provider_name="maisaka_builtin")
             for name in ("web_search", "web_fetch", "reply")]
    async def invoke(invocation, context, **kwargs):
        invoked.append(invocation)
        return ToolExecutionResult(invocation.tool_name, True, content="source result:" + invocation.tool_name)
    runtime = SimpleNamespace(
        session_id="real-chat", _chat_history=[], _chat_loop_service=SimpleNamespace(passive_replyer_override=None, _build_chat_system_prompt=lambda: "Stable persona"),
        _tool_registry=SimpleNamespace(list_tools=AsyncMock(return_value=specs), invoke=invoke),
        set_current_action_tool_names=lambda names: None, _start_planner_continuation=lambda: None,
        _end_planner_continuation=lambda: None, _agent_state="idle", _STATE_RUNNING="running",
        _enter_stop_state=lambda: None, _update_stage_status=lambda *args: None,
        _running=False, _schedule_message_turn=lambda: None, _has_pending_messages=lambda: False,
        expand_message_id_aliases=lambda args: args, _trigger_trimmed_history_learning=AsyncMock(),
    )
    obj = SimpleNamespace(_runtime=runtime, _build_tool_availability_context=lambda: None,
                          _build_tool_execution_context=lambda _: None, _record_tool_execution_effects=AsyncMock(),
                          _build_tool_result_media_messages=lambda call, result: [])
    return obj, invoked


class ScriptedProvider:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.payloads = []
        self.count = 0

    async def __call__(self, params):
        self.payloads.append(context_from_pi(params["messages"]))
        self.count += 1
        call = next(self.calls, None)
        meta = ContextItemMeta.create(logical_turn_id=f"native-turn-{self.count}")
        if call:
            name, args = call
            item = FunctionCallItem(meta, ContextToolCall.create(call_id=f"call-{self.count}", func_name=name, args=args))
            content = [{"type": "toolCall", "id": item.tool_call.call_id, "name": name, "arguments": args}]
        else:
            item = AssistantMessageItem(meta, (ContextTextPart("done"),))
            content = [{"type": "text", "text": "done"}]
        return {"message": {"role": "assistant", "content": content, "stopReason": "toolUse" if call else "stop",
                            "native": {"items": encode_items([item])}}}


@pytest.mark.asyncio
async def test_real_pi_search_read_reply_and_next_batch_prefix(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, invoked = engine()
    host._runtime._chat_history.append(history("first", "m1"))
    runner = PiGroupRunner(host, data_dir=tmp_path)
    scripted = ScriptedProvider([("web_search", {}), ("web_fetch", {}), ("reply", {"text": "exact main agent text"}), None])
    runner.provider = scripted
    try:
        await runner.run([])
        assert [item.tool_name for item in invoked] == ["web_search", "web_fetch", "reply"]
        assert invoked[-1].arguments["text"] == "exact main agent text"
        before = await runner.bridge.request("state", sessionId=runner.session_id)
        snapshot = await runner._db("snapshot", runner.session_id)
        assert [event.payload for event in snapshot.events] == before["messages"]
        host._runtime._chat_history.append(history("second", "m2"))
        await runner.run([])
        after = await runner.bridge.request("state", sessionId=runner.session_id)
        assert after["messages"][:len(before["messages"])] == before["messages"]
        assert scripted.payloads[-1][:len(context_from_pi(before["messages"]))] == context_from_pi(before["messages"])
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_ignored_input_is_durable_and_restart_does_not_duplicate(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    host._runtime._chat_history.append(history("ignored by Luna", "ignored"))
    first = PiGroupRunner(host, data_dir=tmp_path)
    await first.ingest()
    await first.close()
    second = PiGroupRunner(host, data_dir=tmp_path)
    scripted = ScriptedProvider([])
    second.provider = scripted
    try:
        await second.run([])
        payload = scripted.payloads[0]
        assert len([item for item in payload if item.meta.item_id == "ignored"]) == 1
        with pytest.raises(PermissionError):
            await second._callback("model", {"sessionId": "other-chat", "messages": []})
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_references_are_delta_and_restore_exact_projection(monkeypatch, tmp_path: Path):
    from src.maisaka.pi_runtime.references import ReferenceRecord
    from src.maisaka.pi_runtime.provider import ledger_event_message
    reference = ReferenceRecord("memory", "source-1", "v1", "original evidence")
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[reference]))
    host, _ = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    runner.provider = ScriptedProvider([])
    try:
        await runner.run([])
        before = await runner.bridge.request("state", sessionId=runner.session_id)
        await runner.run([])
        snapshot = await runner._db("snapshot", runner.session_id)
        assert len([event for event in snapshot.events if event.kind == "reference"]) == 1
        state = await runner.bridge.request("state", sessionId=runner.session_id)
        assert state["messages"][:len(before["messages"])] == before["messages"]
        assert [ledger_event_message(event) for event in snapshot.events] == state["messages"]
    finally:
        await runner.close()
    restored = PiGroupRunner(host, data_dir=tmp_path)
    try:
        await restored.ensure_initialized()
        state2 = await restored.bridge.request("state", sessionId=restored.session_id)
        assert state2["messages"] == state["messages"]
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_child_transcript_isolated_readable_and_no_parent_write(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, invoked = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    runner.provider = ScriptedProvider([("subagent", {"task": "verify", "tools": ["web_search", "reply"]}), None])
    runner.child_provider = ScriptedProvider([("web_search", {}), None])
    try:
        await runner.run([])
        assert [item.tool_name for item in invoked] == ["web_search"]
        snapshot = await runner._db("snapshot", runner.session_id)
        parent_messages = [event.payload for event in snapshot.events]
        assert not any(message.get("toolName") == "web_search" for message in parent_messages)
        receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
        import json
        delegation = json.loads(receipt.result["content"][0]["text"])
        detail = await runner.subagents.read(runner.session_id, delegation["handle"])
        assert "web_search" in str(detail)
        with pytest.raises(PermissionError):
            await runner.subagents.read("another-group", delegation["handle"])
    finally:
        await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("known", [True, False])
async def test_restart_never_reexecutes_lost_tool_ack(monkeypatch, tmp_path: Path, known):
    from src.llm_models.payload_content.context_item import FunctionCallOutputItem
    from src.maisaka.pi_runtime.bridge import PiBridgeError
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, invoked = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    await runner.ensure_initialized()
    item = FunctionCallItem(ContextItemMeta.create(logical_turn_id="original-turn"), ContextToolCall.create(call_id="lost", func_name="reply", args={"text": "already sent"}))
    assistant = {"role": "assistant", "content": [{"type": "toolCall", "id": "lost", "name": "reply", "arguments": {"text": "already sent"}}], "stopReason": "toolUse", "native": {"items": encode_items([item])}}
    await runner._db("append_event", runner.session_id, "lost-assistant", assistant)
    await runner._db("prepare_tool_execution", runner.session_id, "lost", {"name": "reply", "arguments": {"text": "already sent"}})
    await runner._db("claim_tool_execution", runner.session_id, "lost")
    if known:
        output = FunctionCallOutputItem(ContextItemMeta.create(logical_turn_id="original-turn"), "lost", "sent receipt", "reply")
        await runner._db("record_tool_outcome", runner.session_id, "lost", {"content": [{"type": "text", "text": "sent receipt"}], "details": {"native": {"items": encode_items([output])}}, "isError": False, "terminate": False})
    await runner.close()
    recovered = PiGroupRunner(host, data_dir=tmp_path)
    recovered.provider = ScriptedProvider([])
    try:
        if known:
            await recovered.run([])
            state = await recovered.bridge.request("state", sessionId=recovered.session_id)
            assert any(message.get("toolCallId") == "lost" for message in state["messages"])
        else:
            with pytest.raises(PiBridgeError, match="UNRESOLVED_TOOL_CALLS"):
                await recovered.run([])
        assert invoked == []
    finally:
        await recovered.close()


@pytest.mark.asyncio
async def test_unknown_side_effect_blocks_following_tool_and_provider(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    attempts = []
    async def uncertain(invocation, context, **kwargs):
        attempts.append(invocation.call_id)
        raise OSError("connection lost after send")
    host._runtime._tool_registry.invoke = uncertain
    runner = PiGroupRunner(host, data_dir=tmp_path)
    model_calls = []
    async def model(params):
        model_calls.append(params)
        content, items = [], []
        for call_id in ("first-send", "second-send"):
            item = FunctionCallItem(ContextItemMeta.create(logical_turn_id="batch"), ContextToolCall.create(call_id=call_id, func_name="reply", args={"text": "sent?"}))
            items.append(item)
            content.append({"type": "toolCall", "id": call_id, "name": "reply", "arguments": {"text": "sent?"}})
        return {"message": {"role": "assistant", "content": content, "stopReason": "toolUse", "native": {"items": encode_items(items)}}}
    runner.provider = model
    try:
        with pytest.raises(RuntimeError, match="paused"):
            await runner.run([])
        assert attempts == ["first-send"]
        assert len(model_calls) == 1
        receipt = await runner._db("lookup_tool_execution", runner.session_id, "first-send")
        assert receipt.status == "unknown"
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_tool_media_and_arrivals_append_after_matching_result(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    pending = []
    host._runtime._has_pending_messages = lambda: bool(pending)
    async def ingest_pending():
        host._runtime._chat_history.extend(pending)
        pending.clear()
    host._ingest_pending_messages = ingest_pending
    async def invoke(invocation, context, **kwargs):
        pending.append(history("arrived during tool", "arrival"))
        return ToolExecutionResult(invocation.tool_name, True, content="image receipt", post_history_messages=[history("image reference", "image")])
    host._runtime._tool_registry.invoke = invoke
    runner = PiGroupRunner(host, data_dir=tmp_path)
    scripted = ScriptedProvider([("web_fetch", {}), None])
    runner.provider = scripted
    try:
        await runner.run([])
        state = await runner.bridge.request("state", sessionId=runner.session_id)
        messages = state["messages"]
        receipt_index = next(i for i, message in enumerate(messages) if message.get("toolCallId") == "call-1")
        arrival_index = next(i for i, message in enumerate(messages) if message.get("hostEventId") == "input:arrival")
        image_index = next(i for i, message in enumerate(messages) if message.get("hostEventId") == "input:image")
        assert receipt_index < arrival_index and receipt_index < image_index
        assert not pending
        snapshot = await runner._db("snapshot", runner.session_id)
        assert [event.payload for event in snapshot.events] == messages
    finally:
        await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tool", [True, False])
async def test_maintenance_compacts_then_resumes_only_unfinished_tool_loop(monkeypatch, tmp_path: Path, with_tool):
    from src.maisaka.pi_runtime.provider import ledger_event_message
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    host._runtime._chat_history.append(history("old conversation", "older"))
    runner = PiGroupRunner(host, data_dir=tmp_path)
    scripted = ScriptedProvider([("web_search", {}) if with_tool else None, None])
    async def pressured(params):
        response = await scripted(params)
        response["message"]["hostMaintenanceRequired"] = scripted.count == 1
        return response
    runner.provider = pressured
    runner.summarize = AsyncMock(return_value="Old conversation; preserve the open search task and source IDs.")
    try:
        await runner.run([])
        assert scripted.count == (2 if with_tool else 1)
        assert runner.summarize.await_count == 1
        state = await runner.bridge.request("state", sessionId=runner.session_id)
        assert not state["maintenanceRequired"]
        snapshot = await runner._db("snapshot", runner.session_id)
        assert [ledger_event_message(event) for event in snapshot.events] == state["messages"]
        assert any(event.kind == "summary" for event in snapshot.events)
        host._runtime._trigger_trimmed_history_learning.assert_awaited_once()
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_terminating_tool_does_not_consume_later_host_input(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    pending = []
    host._runtime._has_pending_messages = lambda: bool(pending)
    host._ingest_pending_messages = AsyncMock(side_effect=AssertionError("terminated turn must not drain incoming queue"))
    async def invoke(invocation, context, **kwargs):
        pending.append(history("arrived while ending", "later"))
        return ToolExecutionResult(invocation.tool_name, True, content="stopped", stop_after_execution=True)
    host._runtime._tool_registry.invoke = invoke
    runner = PiGroupRunner(host, data_dir=tmp_path)
    scripted = ScriptedProvider([("reply", {"text": "done"}), None])
    runner.provider = scripted
    try:
        await runner.run([])
        assert len(pending) == 1
        assert scripted.count == 1
        host._ingest_pending_messages.assert_not_awaited()
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_inbox_survives_crash_before_luna_and_no_node_needed(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    host._build_history_message = AsyncMock(return_value=history("arrived before cooldown", "random-render-id"))
    incoming = SimpleNamespace(message_id="platform-source-1")
    first = PiGroupRunner(host, data_dir=tmp_path)
    await first.record_pending(incoming)
    await first.record_pending(incoming)
    assert not first._initialized
    assert await first.pending_source_ids() == ["platform-source-1"]
    assert not (await first._db("snapshot", first.session_id)).events
    await first.close()
    second = PiGroupRunner(host, data_dir=tmp_path)
    scripted = ScriptedProvider([])
    second.provider = scripted
    try:
        await second.run([])
        assert any(item.meta.item_id == "input:source:platform-source-1:0" for item in scripted.payloads[0])
        assert await second.pending_source_ids() == []
        await second.record_pending(incoming)
        assert await second.pending_source_ids() == []
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_actual_web_provider_specs_are_allowed_for_child():
    from src.web_search.provider import WebSearchToolProvider
    from src.maisaka.pi_runtime.runner import child_tool_allowed, pi_tool
    config = SimpleNamespace(enabled=True, search="search", fetch="fetch", services={"search": SimpleNamespace(adapter="anysearch")})
    provider = WebSearchToolProvider(None, lambda: config)
    specs = await provider.list_tools()
    assert {spec.name for spec in specs} == {"web_search", "web_fetch"}
    assert all(child_tool_allowed(spec) for spec in specs)
    assert pi_tool(specs[0].to_llm_definition())["parameters"]["required"] == ["query"]
    specs[0].annotation.read_only = False
    assert not child_tool_allowed(specs[0])


def test_actual_reply_schema_keeps_required_fields():
    from src.maisaka.builtin_tool.reply import get_tool_spec
    from src.maisaka.pi_runtime.runner import pi_tool
    spec = get_tool_spec()
    converted = pi_tool(spec.to_llm_definition())
    assert converted["parameters"]["required"] == ["msg_id", "text"]
    assert converted["parameters"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_transient_provider_error_can_recover_without_rewriting_history(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    async def broken(params):
        raise ConnectionError("offline")
    runner.provider = broken
    try:
        with pytest.raises(RuntimeError, match="paused"):
            await runner.run([])
        before = await runner.bridge.request("state", sessionId=runner.session_id)
        assert before["messages"][-1]["hostSynthetic"] == "provider_callback_failure"
        runner.provider = ScriptedProvider([])
        await runner.run([])
        after = await runner.bridge.request("state", sessionId=runner.session_id)
        assert after["messages"][:len(before["messages"])] == before["messages"]
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_repeated_known_overlimit_is_bounded_not_infinite_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    async def overlimit(params):
        return {"message": {"role": "assistant", "content": [], "stopReason": "stop", "native": {"items": []},
                            "hostControl": "capacity_admission", "hostMaintenanceRequired": True, "hostContinuationRequired": True}}
    runner.provider = overlimit
    runner.summarize = AsyncMock(return_value="Already compact summary")
    try:
        with pytest.raises(RuntimeError, match="two explicit compactions"):
            await runner.run([])
        assert runner.summarize.await_count == 2
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_runner_promotes_raw_evidence_and_managed_read(monkeypatch, tmp_path: Path):
    from src.maisaka.pi_runtime.evidence import DurableEvidenceStore
    from src.maisaka.tool_result_store import ToolResultStore
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    host._runtime._tool_registry.list_tools.return_value.append(ToolSpec(name="read_tool_result", description="read", parameters_schema={"type": "object", "additionalProperties": True}, provider_name="maisaka_builtin"))
    runner = PiGroupRunner(host, data_dir=tmp_path)
    await runner.ensure_initialized()
    store = ToolResultStore()
    runner.evidence = DurableEvidenceStore(tmp_path / "raw-evidence.sqlite3", store)
    read_args = {}
    async def invoke(invocation, context, **kwargs):
        handle = store.save(runner.session_id, "web_fetch", "original source detail, not just preview")
        read_args["result_id"] = handle
        return ToolExecutionResult("web_fetch", True, content="short preview", metadata={"result_id": handle})
    host._runtime._tool_registry.invoke = invoke
    runner.provider = ScriptedProvider([("web_fetch", {}), ("read_tool_result", read_args), None])
    try:
        await runner.run([])
        receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-2")
        assert "original source detail" in receipt.result["content"][0]["text"]
        assert receipt.status == "succeeded"
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_bad_local_evidence_handle_is_known_failure_not_effect_fence(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    scripted = ScriptedProvider([("read_subagent", {"handle": "wrong-handle"}), None])
    runner.provider = scripted
    try:
        await runner.run([])
        receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
        assert receipt.status == "failed"
        assert receipt.result["isError"] is True
        assert scripted.count == 2
        await runner._db("assert_no_unresolved_executions", runner.session_id)
    finally:
        await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["schema", "provider"])
async def test_restored_tool_contract_change_fails_closed(tmp_path: Path, change):
    host, invoked = engine()
    first = PiGroupRunner(host, data_dir=tmp_path)
    await first.ensure_initialized()
    await first.close()
    spec = host._runtime._tool_registry.list_tools.return_value[0]
    if change == "schema":
        spec.parameters_schema = {"type": "object", "required": ["new_argument"]}
    else:
        spec.provider_name = "different-owner"
    restored = PiGroupRunner(host, data_dir=tmp_path)
    try:
        with pytest.raises(RuntimeError, match="contract changed"):
            await restored.ensure_initialized()
        assert invoked == []
    finally:
        await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["schema", "provider", "removed"])
async def test_hot_tool_rebinding_rejected_before_effect_without_unknown_fence(monkeypatch, tmp_path: Path, change):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, invoked = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    await runner.ensure_initialized()
    specs = host._runtime._tool_registry.list_tools.return_value
    if change == "schema":
        specs[0].parameters_schema = {"type": "object", "required": ["changed"]}
    elif change == "provider":
        specs[0].provider_name = "takeover-provider"
    else:
        specs.pop(0)
    scripted = ScriptedProvider([("web_search", {}), None])
    runner.provider = scripted
    try:
        await runner.run([])
        assert invoked == []
        assert scripted.count == 2
        assert await runner._db("lookup_tool_execution", runner.session_id, "call-1") is None
        await runner._db("assert_no_unresolved_executions", runner.session_id)
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_new_tool_declaration_appends_binding_without_changing_prefix(tmp_path: Path):
    host, _ = engine()
    first = PiGroupRunner(host, data_dir=tmp_path)
    await first.ensure_initialized()
    before = await first.bridge.request("state", sessionId=first.session_id)
    await first.close()
    host._runtime._tool_registry.list_tools.return_value.append(ToolSpec(name="new_tool", description="new", parameters_schema={"type": "object"}, provider_name="new-provider", provider_type="plugin"))
    second = PiGroupRunner(host, data_dir=tmp_path)
    try:
        await second.ensure_initialized()
        after = await second.bridge.request("state", sessionId=second.session_id)
        assert after["messages"][:len(before["messages"])] == before["messages"]
        assert after["messages"][-1]["toolBindings"]["new_tool"]["provider_name"] == "new-provider"
        assert after["messages"][-1]["toolBindings"]["new_tool"]["provider_type"] == "plugin"
    finally:
        await second.close()
    third = PiGroupRunner(host, data_dir=tmp_path)
    try:
        await third.ensure_initialized()
    finally:
        await third.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "partial_send"])
async def test_actual_registry_failure_after_mutation_fences_thread(monkeypatch, tmp_path: Path, failure):
    from src.core.tooling import ToolRegistry
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    calls = []
    class Provider:
        provider_name = "maisaka_builtin"
        provider_type = "builtin"
        async def list_tools(self, context):
            return [ToolSpec(name="reply", description="reply", parameters_schema={"type": "object"}, provider_name=self.provider_name, provider_type=self.provider_type)]
        async def invoke(self, invocation, context):
            calls.append(invocation.call_id)
            if failure == "exception":
                raise OSError("transport failed after send")
            return ToolExecutionResult("reply", False, error_message="second segment failed", metadata={"sent_message_ids": ["sent-1"]})
    registry = ToolRegistry()
    registry.register_provider(Provider())
    host._runtime._tool_registry = registry
    runner = PiGroupRunner(host, data_dir=tmp_path)
    scripted = ScriptedProvider([("reply", {"text": "message"}), None])
    runner.provider = scripted
    try:
        with pytest.raises(RuntimeError, match="paused"):
            await runner.run([])
        assert calls == ["call-1"]
        assert scripted.count == 1
        receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
        assert receipt.status == "unknown"
        if failure == "partial_send":
            assert receipt.result["metadata"]["sent_message_ids"] == ["sent-1"]
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_actual_registry_revalidates_after_runner_check(monkeypatch, tmp_path: Path):
    from src.core.tooling import ToolRegistry
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    host, _ = engine()
    invoked = []
    class Provider:
        provider_name = "maisaka_builtin"
        provider_type = "builtin"
        count = 0
        async def list_tools(self, context):
            self.count += 1
            # Initialization and runner check see the original contract; the
            # actual dispatch route sees a changed schema after those awaits.
            schema = {"type": "object"} if self.count <= 2 else {"type": "object", "required": ["changed"]}
            return [ToolSpec(name="reply", description="reply", parameters_schema=schema, provider_name=self.provider_name, provider_type=self.provider_type)]
        async def invoke(self, invocation, context):
            invoked.append(invocation)
            return ToolExecutionResult("reply", True, content="must not happen")
    registry = ToolRegistry()
    registry.register_provider(Provider())
    host._runtime._tool_registry = registry
    runner = PiGroupRunner(host, data_dir=tmp_path)
    runner.provider = ScriptedProvider([("reply", {"text": "message"}), None])
    try:
        await runner.run([])
        assert not invoked
        receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
        assert receipt.status == "failed"
        assert receipt.result["isError"] is True
        await runner._db("assert_no_unresolved_executions", runner.session_id)
    finally:
        await runner.close()
