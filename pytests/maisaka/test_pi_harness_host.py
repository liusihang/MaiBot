"""Real coding-agent subprocess and local mock wire: no Python model callback."""

from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncio
import json
import re

import pytest

from src.common.data_models.message_component_data_model import MessageSequence
from src.core.tooling import ToolExecutionResult, ToolSpec
from src.maisaka.context.messages import ContentContextMessage
from src.maisaka.pi_runtime.runner import PiGroupRunner


def history(text, identifier):
    return ContentContextMessage(
        raw_message=MessageSequence([]), visible_text=text, timestamp=datetime.now(), context_item_id=identifier
    )


def engine():
    invoked = []
    specs = [
        ToolSpec(
            name=name,
            description=name,
            parameters_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            provider_name="maisaka_builtin",
        )
        for name in ("web_search", "web_fetch", "reply")
    ]

    async def invoke(invocation, context, **kwargs):
        invoked.append(invocation)
        return ToolExecutionResult(invocation.tool_name, True, content="source:" + invocation.tool_name)

    runtime = SimpleNamespace(
        session_id="real-chat",
        _chat_history=[],
        _chat_loop_service=SimpleNamespace(
            passive_replyer_override=None, _build_chat_system_prompt=lambda: "Stable group persona"
        ),
        _tool_registry=SimpleNamespace(list_tools=AsyncMock(return_value=specs), invoke=invoke),
        set_current_action_tool_names=lambda names: None,
        _start_planner_continuation=lambda: None,
        _end_planner_continuation=lambda: None,
        _agent_state="idle",
        _STATE_RUNNING="running",
        _enter_stop_state=lambda: None,
        _update_stage_status=lambda *args: None,
        _running=False,
        _schedule_message_turn=lambda: None,
        _has_pending_messages=lambda: False,
        _arm_forced_turn_state=Mock(),
        _queue_proactive_turn=Mock(),
        _build_proactive_trigger_message=Mock(),
        _trigger_pi_source_learning=AsyncMock(),
        expand_message_id_aliases=lambda args: args,
    )
    return SimpleNamespace(
        _runtime=runtime,
        _build_tool_availability_context=lambda: None,
        _build_tool_execution_context=lambda _: None,
        _record_tool_execution_effects=AsyncMock(),
        _build_tool_result_media_messages=lambda call, result: [],
    ), invoked


@contextmanager
def wire_server(calls=(), *, prompt_tokens=10):
    requests = []
    scripted = iter(calls)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            call = next(scripted, None)
            if callable(call):
                call = call(request, requests)
            delta = {"content": call["text"] if isinstance(call, dict) else ""}
            finish = "stop"
            if call and not isinstance(call, dict):
                name, arguments = call
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-" + str(len(requests)),
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ]
                }
                finish = "tool_calls"
            data = [
                {
                    "id": "answer",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "mock-model",
                    "choices": [{"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": None}],
                },
                {
                    "id": "answer",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "mock-model",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 2,
                        "total_tokens": prompt_tokens + 2,
                    },
                },
            ]
            body = ("".join("data: " + json.dumps(item) + "\n\n" for item in data) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    config = {
        "api_providers": [
            {
                "name": "mock",
                "client_type": "openai",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "api_key": "test",
            }
        ],
        "models": [
            {
                "name": "mock",
                "model_type": "llm",
                "model_identifier": "mock-model",
                "api_provider": "mock",
                "max_tokens": 1000,
            }
        ],
        "model_task_config": {name: {"model_list": ["mock"], "max_tokens": 1000} for name in ("planner",)},
    }
    try:
        yield config, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.asyncio
async def test_chat_plain_draft_is_corrected_by_same_agent_before_host_delivery(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([{"text": "draft not sent"}, ("reply", {"text": "visible"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        host._runtime._chat_history.append(history("question", "m1"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert [item.tool_name for item in invoked] == ["reply"]
            assert invoked[0].arguments["text"] == "visible"
            assert "Your preceding text was not sent" in json.dumps(requests[1]["messages"])
            assert len(requests) == 3
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_host_finished_reply_uses_only_one_real_sdk_model_request(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("reply", {"text": "one answer"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, effects = engine()
        next(spec for spec in host._runtime._tool_registry.list_tools.return_value if spec.name == "reply").provider_type = "builtin"

        async def invoke(invocation, context, **kwargs):
            effects.append(invocation)
            return ToolExecutionResult("reply", True, content="sent", metadata={"finish_turn": True})

        host._runtime._tool_registry.invoke = invoke
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert len(requests) == 1
            assert len(effects) == 1
            assert runner.ledger.pending_turn(runner.session_id) is None
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_host_admits_a_late_answer_at_stop_and_continues_the_same_thread(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("stop", {}), ("reply", {"text": "got the answer"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, effects = engine()
        next(spec for spec in host._runtime._tool_registry.list_tools.return_value if spec.name == "reply").provider_type = "builtin"
        host._runtime._tool_registry.list_tools.return_value.append(
            ToolSpec(name="stop", description="stop", parameters_schema={"type": "object"},
                     provider_name="maisaka_builtin", provider_type="builtin")
        )
        pending = False
        source = original_source_message()
        host._runtime._has_pending_messages = lambda: pending
        host._runtime._enter_running_state = Mock()

        async def ingest_pending():
            nonlocal pending
            pending = False
            host._runtime._chat_history.append(history("the answer you just asked for", "late-answer"))
            return [source]

        host._ingest_pending_messages = AsyncMock(side_effect=ingest_pending)

        async def invoke(invocation, context, **kwargs):
            nonlocal pending
            effects.append(invocation)
            if invocation.tool_name == "stop":
                pending = True
                return ToolExecutionResult("stop", True, content="stopped", stop_after_execution=True)
            return ToolExecutionResult("reply", True, content="sent", metadata={"finish_turn": True})

        host._runtime._tool_registry.invoke = invoke
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert len(requests) == 2
            assert [effect.tool_name for effect in effects] == ["stop", "reply"]
            assert "the answer you just asked for" in json.dumps(requests[1])
            host._runtime._enter_running_state.assert_called_once()
            assert runner.ledger.pending_turn(runner.session_id) is None
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_inbound_self_echo_is_typed_by_sender_identity(monkeypatch):
    from src.common.data_models.message_component_data_model import TextComponent
    from src.maisaka.context.messages import SessionBackedMessage
    from src.maisaka.pi_runtime.runner import history_message

    source = original_source_message()
    source.raw_message = MessageSequence([TextComponent("own message")])
    history_entry = SessionBackedMessage.from_session_message(
        source, raw_message=source.raw_message, visible_text="own message"
    )
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.is_bot_self", lambda platform, user_id: True)
    assert (await history_message(history_entry, "echo"))["hostSelfEcho"] is True
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.is_bot_self", lambda platform, user_id: False)
    assert "hostSelfEcho" not in await history_message(history_entry, "other-user")


@pytest.mark.asyncio
async def test_real_harness_search_read_reply_stable_wire_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("web_search", {}), ("web_fetch", {}), ("reply", {"text": "main exact text"})]) as (
        config,
        requests,
    ):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        host._runtime._chat_history.append(history("first", "m1"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert [item.tool_name for item in invoked] == ["web_search", "web_fetch", "reply"]
            assert invoked[-1].arguments["text"] == "main exact text"
            before = requests[-1]["messages"]
            host._runtime._chat_history.append(history("second", "m2"))
            await runner.run([])
            assert requests[-1]["messages"][: len(before)] == before
            assert not hasattr(runner.ledger, "snapshot")
            assert not hasattr(runner, "provider")
            with pytest.raises(ValueError, match="does not implement model"):
                await runner._callback("model", {"sessionId": runner.session_id})
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_ignored_messages_persist_in_real_sdk_and_restart_once(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server() as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        host._runtime._chat_history.append(history("ignored unique message", "ignored"))
        first = PiGroupRunner(host, data_dir=tmp_path)
        await first.ingest()
        assert not requests
        await first.close()
        second = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await second.ensure_initialized()
            host._runtime._queue_proactive_turn.assert_not_called()
            await second.run([])
            wire = json.dumps(requests[0]["messages"])
            assert wire.count("ignored unique message") == 1
        finally:
            await second.close()


@pytest.mark.asyncio
async def test_evidence_failure_after_confirmed_send_preserves_success(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("reply", {"text": "confirmed message"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        next(spec for spec in host._runtime._tool_registry.list_tools.return_value if spec.name == "reply").provider_type = "builtin"
        host._runtime._tool_registry.invoke = AsyncMock(return_value=ToolExecutionResult(
            "reply", True, content="message sent", metadata={"finish_turn": True, "sent_message_ids": ["sent-1"]}
        ))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.ensure_initialized()
            monkeypatch.setattr(runner.evidence, "capture", AsyncMock(side_effect=OSError("evidence unavailable")))
            await runner.run([])
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            assert receipt.status == "succeeded"
            assert receipt.result["isError"] is False
            assert receipt.result["details"]["sent_message_ids"] == ["sent-1"]
            assert len(requests) == 1
            host._runtime._tool_registry.invoke.assert_awaited_once()
        finally:
            await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "partial_send"])
async def test_unknown_send_blocks_only_same_effect_and_allows_chat_after_restart(monkeypatch, tmp_path, failure):
    from src.core.tooling import ToolRegistry

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    effects = []

    class Provider:
        provider_name = "maisaka_builtin"
        provider_type = "builtin"

        async def list_tools(self, context):
            return [ToolSpec(name=name, description=name, parameters_schema={"type": "object"},
                             provider_name=self.provider_name, provider_type=self.provider_type)
                    for name in ("reply", "web_fetch")]

        async def invoke(self, invocation, context):
            effects.append(invocation)
            if invocation.tool_name == "reply" and invocation.arguments["text"] == "command":
                if failure == "exception":
                    raise OSError("acknowledgement lost after sending")
                return ToolExecutionResult("reply", False, error_message="second segment failed",
                                           metadata={"sent_message_ids": ["sent-1"]})
            return ToolExecutionResult(invocation.tool_name, True, content="ok",
                                       metadata={"finish_turn": invocation.tool_name == "reply"})

    original = {"msg_id": "old-question", "text": "command"}
    with wire_server([
        ("reply", original),
        ("reply", {**original, "finish_turn": False}),
        ("web_fetch", {"url": "https://example.com"}),
        ("reply", {"msg_id": "new-question", "text": "continue chatting"}),
        ("reply", {"msg_id": "later-question", "text": "after restart"}),
    ]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        registry = ToolRegistry()
        registry.register_provider(Provider())
        host._runtime._tool_registry = registry
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert len(requests) == 4
            assert [item.tool_name for item in effects] == ["reply", "web_fetch", "reply"]
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            assert receipt.status == "unknown"
            assert receipt.result["isError"] is True
            assert receipt.result["details"]["execution_outcome"] == "unknown"
            if failure == "partial_send":
                assert "sent-1" in str(receipt.result)
            duplicate = await runner._db("lookup_tool_execution", runner.session_id, "call-2")
            assert duplicate.status == "failed"
            assert duplicate.result["details"]["execution_outcome"] == "not_executed"
            if failure == "exception":
                assert "acknowledgement lost" in json.dumps(requests[1])
        finally:
            await runner.close()
        from pathlib import Path
        import subprocess

        session_file = next((tmp_path / "sessions").glob("*.jsonl"))
        append_legacy_marker = """import {SessionManager} from '@earendil-works/pi-coding-agent';
import {dirname} from 'node:path';
const file=process.argv[1];
SessionManager.open(file,dirname(file),process.cwd()).appendCustomEntry(
  'maibot.tool_unknown',{callId:'call-1',name:'reply'});"""
        process = await asyncio.to_thread(
            subprocess.run, ["node", "--input-type=module", "-e", append_legacy_marker, str(session_file)],
            cwd=Path(__file__).resolve().parents[2] / "runtime/pi_agent", capture_output=True, text=True,
        )
        assert process.returncode == 0, process.stderr
        restored = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await restored.run([])
            assert len(requests) == 5
            assert effects[-1].arguments["text"] == "after restart"
            assert len(effects) == 4
            state = await restored.bridge.request("state", sessionId=restored.session_id)
            assert not state["paused"] and state["error"] is None
        finally:
            await restored.close()


def original_source_message():
    from src.chat.message_receive.message import SessionMessage
    from src.common.data_models.mai_message_data_model import GroupInfo, MessageInfo, UserInfo
    from src.common.data_models.message_component_data_model import (
        AtComponent,
        ImageComponent,
        ReplyComponent,
        TextComponent,
    )

    message = SessionMessage(message_id="source-immutable", timestamp=datetime(2026, 1, 2), platform="qq")
    message.session_id = "real-chat"
    message.message_info = MessageInfo(UserInfo("sender", "old alias"), GroupInfo("group-id", "old group"))
    message.raw_message = MessageSequence(
        [
            TextComponent("original text"),
            ImageComponent(binary_hash="image-hash", content="old generated image description", binary_data=b"fixture"),
            AtComponent("target", "old target alias"),
            ReplyComponent("quoted-message", target_message_content="old quote preview"),
        ]
    )
    return message


@pytest.mark.asyncio
async def test_record_pending_uses_original_source_identity_not_mutable_rendering(monkeypatch, tmp_path):
    from copy import deepcopy

    from src.maisaka.pi_runtime.business_store import LedgerConflict

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server() as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        host._build_history_message = AsyncMock(return_value=history("frozen original source", "render-id"))
        source = original_source_message()
        runner = PiGroupRunner(host, data_dir=tmp_path)
        await runner.record_pending(source)
        await runner.record_pending(source)
        assert runner.bridge._process is None
        assert await runner.pending_source_ids() == [source.message_id]
        assert host._build_history_message.await_count == 1

        derived = deepcopy(source)
        derived.message_info.user_info.user_nickname = "new alias"
        derived.message_info.group_info.group_name = "new group name"
        derived.raw_message.components[1].content = "new generated image description"
        derived.raw_message.components[2].target_user_nickname = "new target alias"
        derived.raw_message.components[3].target_message_content = "new quote preview"
        derived.received_at = datetime.now()
        await runner.record_pending(derived)
        assert host._build_history_message.await_count == 1
        mutations = [
            lambda item: setattr(item.raw_message.components[0], "text", "different original text"),
            lambda item: setattr(item.raw_message.components[1], "binary_hash", "different-image"),
            lambda item: setattr(item.raw_message.components[2], "target_user_id", "another-target"),
            lambda item: setattr(item.raw_message.components[3], "target_message_id", "another-quote"),
            lambda item: setattr(item.message_info.user_info, "user_id", "another-sender"),
        ]
        for mutate in mutations:
            changed = deepcopy(source)
            mutate(changed)
            with pytest.raises(LedgerConflict, match="original content"):
                await runner.record_pending(changed)
        await runner.close()
        restored = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await restored.run([])
            assert json.dumps(requests[0]["messages"]).count("frozen original source") == 1
            await restored.record_pending(derived)
            assert await restored.pending_source_ids() == []
            assert host._build_history_message.await_count == 1
        finally:
            await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lose_ack", [False, True])
async def test_generic_emoji_media_receipt_survives_lost_ack_and_restart(monkeypatch, tmp_path, lose_ack):
    from src.llm_models.payload_content.context_item import (
        ContextImagePart,
        ContextItemMeta,
        ContextTextPart,
        UserMessageItem,
    )
    from src.maisaka.context.emoji_candidates import EmojiCandidateMessage

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    with wire_server([("web_fetch", {})]) as (config, requests):
        config["models"][0]["visual"] = True
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, effects = engine()

        async def invoke(invocation, context, **kwargs):
            effects.append(invocation)
            candidate = EmojiCandidateMessage(
                item=UserMessageItem(
                    ContextItemMeta.create(), (ContextTextPart("durable collage"), ContextImagePart("png", image_data))
                ),
                emoji_hashes={1000000000: "emoji-hash"},
                visible_text="durable collage",
                timestamp=datetime.now(),
                session_id=host._runtime.session_id,
            )
            return ToolExecutionResult("web_fetch", True, content="collage receipt", post_history_messages=[candidate])

        host._runtime._tool_registry.invoke = invoke
        runner = PiGroupRunner(host, data_dir=tmp_path)
        original_send = runner.bridge._send
        lost = False

        async def lose_completed_tool_ack(payload):
            nonlocal lost
            result = payload.get("result", {})
            if not lost and isinstance(result, dict) and result.get("details", {}).get("hostReceipt"):
                lost = True
                raise OSError("fixture acknowledgement lost after durable receipt")
            await original_send(payload)

        if lose_ack:
            monkeypatch.setattr(runner.bridge, "_send", lose_completed_tool_ack)
        try:
            await runner.run([])
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            assert receipt.status == "succeeded"
            assert receipt.result["content"][-1] == {"type": "image", "data": image_data, "mimeType": "image/png"}
            assert len(effects) == 1 and len(requests) == 2
            before = requests[-1]["messages"]
        finally:
            await runner.close()
        restored = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await restored.run([])
            assert requests[-1]["messages"][: len(before)] == before
            assert len(effects) == 1
            assert json.dumps(requests[-1]["messages"]).count(image_data) == 1
            state = await restored.bridge.request("state", sessionId=restored.session_id)
            assert state["unresolvedToolCalls"] == []
            entries = [
                json.loads(line)
                for file in (tmp_path / "sessions").rglob("*.jsonl")
                for line in file.read_text().splitlines()
            ]
            results = [
                entry["message"]
                for entry in entries
                if entry.get("type") == "message"
                and entry.get("message", {}).get("role") == "toolResult"
                and entry["message"].get("toolCallId") == "call-1"
            ]
            image_part = {"type": "image", "data": image_data, "mimeType": "image/png"}
            # A read-only receipt lookup recovers the lost IPC ACK immediately;
            # the original tool is not re-executed and its image appears once.
            assert len(results) == 1 and image_part in results[0]["content"]
            assert results[0]["isError"] is False
        finally:
            await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("child_tool", ["web_fetch", "reply", "gpt_web_search"])
async def test_actual_child_has_isolated_harness_and_durable_scoped_source_reads(monkeypatch, tmp_path, child_tool):
    from src.maisaka.pi_runtime.evidence import DurableEvidenceStore
    from src.maisaka.tool_result_store import ToolResultStore

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))

    def read_delegation(request, requests):
        tool_outputs = [message["content"] for message in request["messages"] if message.get("role") == "tool"]
        delegation = next(text for text in tool_outputs if "pi-child-" in text)
        return "read_subagent", {"handle": re.search(r'handle="(pi-child-[^\"]+)"', delegation)[1]}

    with wire_server(
        [
            ("subagent", {"task": "child-only research task"}),
            (child_tool, {}),
            {"text": "child-only source-grounded conclusion"},
            read_delegation,
            None,
        ]
    ) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, effects = engine()
        if child_tool == "gpt_web_search":
            host._runtime._tool_registry.list_tools.return_value.append(
                ToolSpec(name=child_tool, description=child_tool, parameters_schema={"type": "object"}, provider_name="isolated")
            )
        source_store = ToolResultStore()

        async def invoke(invocation, context, **kwargs):
            effects.append(invocation)
            if child_tool == "gpt_web_search":
                return ToolExecutionResult(child_tool, True, content=json.dumps({
                    "status": "ok", "content": "child preview only", "result_id": "ws-original",
                    "reader_tool": "gpt_read_search_result",
                }))
            handle = source_store.save(invocation.session_id, child_tool, "child original evidence only")
            return ToolExecutionResult(child_tool, True, content="child preview only", metadata={"result_id": handle})

        host._runtime._tool_registry.invoke = invoke
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.ensure_initialized()
            runner.evidence = DurableEvidenceStore(tmp_path / "evidence.sqlite3", source_store)
            await runner.run([])
            assert [item.tool_name for item in effects] == [child_tool]
            assert len(requests) == 5
            # Existing business sub-agent requests use the planner task mapping;
            # ModelTaskConfig has no independent "sub_agent" task to invent.
            assert set(config["model_task_config"]) == {"planner"}
            assert requests[1]["model"] == requests[0]["model"] == "mock-model"
            assert requests[1]["tools"] == requests[0]["tools"]
            assert "child-only research task" in json.dumps(requests[1]["messages"])
            assert "工具执行助手" in json.dumps(requests[1]["messages"], ensure_ascii=False)
            assert "Stable group persona" not in json.dumps(requests[1]["messages"])
            assert "当前北京时间" in json.dumps(requests[1]["messages"], ensure_ascii=False)
            assert "child preview only" not in json.dumps(requests[3]["messages"])
            assert "child preview only" in json.dumps(requests[4]["messages"])
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            visible = receipt.result["content"][0]["text"]
            handle = re.search(r'handle="(pi-child-[^\"]+)"', visible)[1]
            assert "child-only source-grounded conclusion" in visible
            assert all(field not in visible for field in ("host_tool_receipt", "call_id", "uncertainty", "evidence_handle"))
            delegation = await asyncio.to_thread(runner.subagents._summary, runner.session_id, handle)
            [source] = delegation["sources"]
            assert source["kind"] == "host_tool_receipt"
            detail = await runner.subagents.read(runner.session_id, delegation["handle"])
            assert "child preview only" in detail["text"]
            if child_tool == "gpt_web_search":
                assert source["result_id"] == "ws-original"
                assert source["reader_tool"] == "gpt_read_search_result"
                assert "ws-original" in detail["text"]
            else:
                raw = await runner.evidence.read(runner.session_id, source["result_id"])
                assert "child original evidence only" in json.dumps(raw)
            with pytest.raises(PermissionError):
                await runner.subagents.read("other-group", delegation["handle"])
            if child_tool != "gpt_web_search":
                with pytest.raises(PermissionError):
                    await runner.evidence.read("other-group", source["result_id"])
        finally:
            await runner.close()
        restored = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await restored._ensure_storage()
            assert (await restored.subagents.read(restored.session_id, delegation["handle"]))["text"] == detail["text"]
            if child_tool != "gpt_web_search":
                assert await restored.evidence.read(restored.session_id, source["result_id"]) == raw
        finally:
            await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, arguments",
    [
        ("read_subagent", {"handle": "invalid-child-handle"}),
        ("read_tool_result", {"result_id": "rdeadbeef"}),
        ("read_thread_archive", {"event_ids": ["invalid-archive-entry"]}),
    ],
)
async def test_bad_evidence_handle_is_known_failed_result_not_unknown_effect(monkeypatch, tmp_path, name, arguments):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([(name, arguments)]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            assert receipt.status == "failed" and receipt.result["isError"] is True
            assert len(requests) == 2 and not invoked
            await runner._db("assert_no_unresolved_executions", runner.session_id)
            assert (await runner.bridge.request("state", sessionId=runner.session_id))["unresolvedToolCalls"] == []
        finally:
            await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword", ["", "顺之利刃"])
async def test_memory_reader_sends_only_public_body_to_the_real_model_wire(monkeypatch, tmp_path, keyword):
    from src.maisaka.tool_result_store import tool_result_store

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    raw = json.dumps({
        "query": "马斯克 顺之利刃 称号", "status": "matched",
        "requested_scope": {"chat_id": "PRIVATE_SESSION"},
        "hits": [{"content": "几十块钱的通行证要什么自行车", "hash": "PRIVATE_HASH", "score": 0.52}],
    }, ensure_ascii=False)
    handle = tool_result_store.save("real-chat", "query_memory", raw)
    with wire_server([("read_tool_result", {"result_id": handle, "keyword": keyword})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner._ensure_storage()
            await runner.evidence.capture("real-chat", SimpleNamespace(metadata={"result_id": handle}))
            await runner.run([])
            visible = "\n".join(str(m.get("content", "")) for m in requests[-1]["messages"] if m["role"] == "tool")
            assert ("通行证" in visible) is (not keyword)
            if keyword:
                assert "没有这个关键词" in visible
            for field in ("PRIVATE_HASH", "PRIVATE_SESSION", "requested_scope", "source_session", "matched", "score"):
                assert field not in visible
            assert not invoked
        finally:
            await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["schema", "provider"])
async def test_real_restored_frozen_tool_contract_rebinding_fails_closed(monkeypatch, tmp_path, change):
    with wire_server() as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        host._runtime._chat_history.append(history("persist before contract check", "one"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        await runner.ingest()
        await runner.close()
        spec = host._runtime._tool_registry.list_tools.return_value[0]
        if change == "schema":
            spec.parameters_schema = {"type": "object", "required": ["new-argument"]}
        else:
            spec.provider_name = "different-owner"
        restored = PiGroupRunner(host, data_dir=tmp_path)
        try:
            with pytest.raises(RuntimeError, match="migration|contract|differ"):
                await restored.ensure_initialized()
            assert not invoked and not requests
        finally:
            await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["schema", "provider", "removed"])
async def test_real_hot_tool_rebinding_rejects_before_any_effect(monkeypatch, tmp_path, change):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("web_search", {})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.ensure_initialized()
            specs = host._runtime._tool_registry.list_tools.return_value
            if change == "schema":
                specs[0].parameters_schema = {"type": "object", "required": ["changed"]}
            elif change == "provider":
                specs[0].provider_name = "different-owner"
            else:
                specs.pop(0)
            await runner.run([])
            assert not invoked and len(requests) == 2
            assert not (await runner.bridge.request("state", sessionId=runner.session_id))["paused"]
            assert await runner._db("lookup_tool_execution", runner.session_id, "call-1") is None
            await runner._db("assert_no_unresolved_executions", runner.session_id)
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_actual_registry_revalidates_contract_after_runner_await(monkeypatch, tmp_path):
    from src.core.tooling import ToolRegistry

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    effects = []

    class Provider:
        provider_name = "maisaka_builtin"
        provider_type = "builtin"
        count = 0

        async def list_tools(self, context):
            self.count += 1
            await asyncio.sleep(0)
            schema = {"type": "object"} if self.count <= 2 else {"type": "object", "required": ["changed"]}
            return [
                ToolSpec(
                    name="reply",
                    description="reply",
                    parameters_schema=schema,
                    provider_name=self.provider_name,
                    provider_type=self.provider_type,
                )
            ]

        async def invoke(self, invocation, context):
            effects.append(invocation)
            return ToolExecutionResult("reply", True, content="must not execute")

    with wire_server([("reply", {"text": "do not send"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        registry = ToolRegistry()
        registry.register_provider(Provider())
        host._runtime._tool_registry = registry
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert not effects and len(requests) == 2
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            assert receipt.status == "failed" and receipt.result["isError"] is True
            await runner._db("assert_no_unresolved_executions", runner.session_id)
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_real_web_provider_and_main_reply_schema():
    from src.maisaka.builtin_tool.reply import get_tool_spec
    from src.maisaka.pi_runtime.runner import pi_tool
    from src.web_search.provider import WebSearchToolProvider

    config = SimpleNamespace(
        enabled=True, search="search", fetch="fetch", services={"search": SimpleNamespace(adapter="anysearch")}
    )
    provider = WebSearchToolProvider(None, lambda: config)
    specs = await provider.list_tools()
    assert {spec.name for spec in specs} == {"web_search", "web_fetch"}
    assert pi_tool(specs[0].to_llm_definition())["parameters"]["required"] == ["query"]
    reply = get_tool_spec()
    converted = pi_tool(reply.to_llm_definition())
    assert converted["parameters"]["required"] == ["msg_id", "text"]
    assert converted["parameters"]["additionalProperties"] is False


def test_child_request_kind_keeps_real_chat_service_model_task_mapping():
    from src.maisaka.chat_loop_service import MaisakaChatLoopService
    from src.config.model_configs import ModelTaskConfig

    service = SimpleNamespace(_model_task_name="planner")
    assert MaisakaChatLoopService._resolve_model_task_name(service, "sub_agent") == "planner"
    assert "sub_agent" not in ModelTaskConfig.model_fields


@pytest.mark.asyncio
async def test_host_explicit_sdk_compaction_and_archive_remain_reachable(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server() as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        host._runtime._chat_history.append(history("original archive source", "source-1"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            before = await runner.bridge.request("state", sessionId=runner.session_id)
            result = await runner.compact(summary="explicit useful summary")
            assert result["compacted"]
            assert result["revision"] > before["revision"]
            archived = await runner.bridge.request(
                "read_archive",
                sessionId=runner.session_id,
                eventIds=[result["compaction"]["archiveHandle"]],
                limit=12000,
            )
            assert "original archive source" in json.dumps(archived)
            await runner.run([])
            wire = json.dumps(requests[-1])
            assert "explicit useful summary" in wire
            assert "original archive source" not in wire
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_restored_idle_session_schedules_maintenance_without_waking_chat(monkeypatch, tmp_path):
    host, _ = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    monkeypatch.setattr(runner, "ensure_initialized", AsyncMock())
    monkeypatch.setattr(runner, "_db", AsyncMock(return_value=None))
    schedule = Mock()
    monkeypatch.setattr(runner, "_schedule_context_maintenance", schedule)
    await runner.restore_pending_turn()
    schedule.assert_called_once()
    host._runtime._queue_proactive_turn.assert_not_called()


@pytest.mark.asyncio
async def test_slow_post_reply_maintenance_continues_during_the_next_main_turn(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    entered, release, interrupted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    host, effects = engine()
    with wire_server([("reply", {"text": "first answer"}), None, ("reply", {"text": "second answer"}), None]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        runner = PiGroupRunner(host, data_dir=tmp_path)

        async def slow_maintenance():
            assert runner.ledger.pending_turn(runner.session_id) is None
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                interrupted.set()
                raise

        monkeypatch.setattr(runner, "_maintain_context", slow_maintenance)
        first = asyncio.create_task(runner.run([]))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            assert first.done(), "Post-reply maintenance still occupies the main run loop"
            await first
            host._runtime._chat_history.append(history("next real question", "next-question"))
            await asyncio.wait_for(runner.run([]), 10)
            assert not interrupted.is_set()
            assert [item.arguments["text"] for item in effects] == ["first answer", "second answer"]
            assert len(requests) == 4
            release.set()
            await runner._maintenance_task
        finally:
            release.set()
            await first
            await runner.close()


@pytest.mark.asyncio
async def test_ingress_and_new_reply_complete_before_real_sdk_summary_preserving_all_new_history(monkeypatch, tmp_path):
    from src.config.config import global_config

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    monkeypatch.setattr(global_config.chat, "context_prompt_token_high_watermark", 51200)
    entered, release = Event(), Event()
    host, effects = engine()
    host._build_history_message = AsyncMock(return_value=history("new question while summarizing", "new-source"))

    def slow_summary(request, requests):
        assert "Summarize this group conversation faithfully" in json.dumps(request)
        entered.set()
        release.wait(20)
        return {"text": "completed old conversation summary"}

    with wire_server([
        ("reply", {"text": "first answer"}), None, slow_summary,
        ("reply", {"text": "new answer"}), None,
    ], prompt_tokens=52000) as (config, requests):
        config["model_task_config"]["memory"] = {"model_list": ["mock"], "max_tokens": 1000}
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host._runtime._chat_history.append(history("original source summarized from fixed snapshot", "old-source"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert await asyncio.to_thread(entered.wait, 10)
            summary_task = runner._maintenance_task
            state = await runner.bridge.request("state", sessionId=runner.session_id)
            assert not state["isCompacting"]
            assert state["backgroundCompaction"]["status"] == "summarizing"
            await asyncio.wait_for(runner.record_pending(original_source_message()), 5)
            assert not summary_task.done()
            assert runner._failed_compaction_head is None
            state = await runner.bridge.request("state", sessionId=runner.session_id)
            assert state["unresolvedToolCalls"] == []
            assert runner.ledger.pending_turn(runner.session_id) is None

            schedule = Mock()
            monkeypatch.setattr(runner, "_schedule_context_maintenance", schedule)
            await asyncio.wait_for(runner.run([]), 10)
            assert not release.is_set(), "Main reply waited for the old summary provider"
            assert [item.arguments["text"] for item in effects] == ["first answer", "new answer"]
            assert "original source summarized from fixed snapshot" in json.dumps(requests[3])
            assert "new question while summarizing" in json.dumps(requests[3])
            assert "completed old conversation summary" not in json.dumps(requests[3])
            before = await runner.bridge.request("state", sessionId=runner.session_id)
            schedule.assert_called_once()
            release.set()
            await asyncio.wait_for(summary_task, 10)
            after = await runner.bridge.request("state", sessionId=runner.session_id)
            assert after["revision"] > before["revision"]
            assert after["backgroundCompaction"] is None
            assert after["contextTokens"] < 51200
            assert len(requests) == 5, "New messages must not restart the summary request"
            assert len(effects) == 2, "Applying a summary must not replay any reply"
            assert runner.ledger.pending_turn(runner.session_id) is None
        finally:
            release.set()
            await runner.close()


@pytest.mark.asyncio
async def test_high_watermark_compacts_after_reply_and_durable_completion(monkeypatch, tmp_path):
    from src.config.config import global_config

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    monkeypatch.setattr(global_config.chat, "context_prompt_token_high_watermark", 51200)
    host, effects = engine()

    def summary_after_delivery(request, requests):
        assert len(effects) == 1 and effects[0].tool_name == "reply"
        assert runner.ledger.pending_turn(runner.session_id) is None
        assert "Summarize this group conversation faithfully" in json.dumps(request)
        return {"text": "real summary"}

    with wire_server([("reply", {"text": "already delivered"}), None, summary_after_delivery], prompt_tokens=52000) as (config, requests):
        config["model_task_config"]["memory"] = {"model_list": ["mock"], "max_tokens": 1000}
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host._runtime._chat_history.append(history("original source for post-reply summary" + "x" * 250000, "source-long"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            await runner._maintenance_task
            assert len(requests) == 3
            state = await runner.bridge.request("state", sessionId=runner.session_id)
            assert state["contextTokens"] < 51200
            assert state["unresolvedToolCalls"] == []
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_post_reply_summary_failure_keeps_the_delivered_turn_complete(monkeypatch, tmp_path):
    from src.config.config import global_config
    from src.maisaka.pi_runtime.bridge import PiBridgeError

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    monkeypatch.setattr(global_config.chat, "context_prompt_token_high_watermark", 51200)
    with wire_server([("reply", {"text": "visible answer"}), None], prompt_tokens=52000) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, effects = engine()
        host._runtime._chat_history.append(history("question", "source"))
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.ensure_initialized()
            original = runner.bridge.request
            failed_calls = []

            async def request(method, **params):
                if method == "start_compaction":
                    assert len(effects) == 1
                    assert runner.ledger.pending_turn(runner.session_id) is None
                    failed_calls.append(params)
                    raise PiBridgeError("Pi summary request did not complete", code="SUMMARY_FAILED")
                return await original(method, **params)

            monkeypatch.setattr(runner.bridge, "request", request)
            await runner.run([])
            await runner._maintenance_task
            assert len(requests) == 2
            assert len(effects) == 1 and effects[0].arguments["text"] == "visible answer"
            assert runner.ledger.pending_turn(runner.session_id) is None
            before = await original("state", sessionId=runner.session_id)
            await runner._maintain_context()
            assert len(failed_calls) == 1
            after = await original("state", sessionId=runner.session_id)
            assert after["archive"] == before["archive"]
            assert after["unresolvedToolCalls"] == []
        finally:
            await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens,should_compact", [(51199, False), (51200, True)])
async def test_maintenance_uses_sdk_token_pressure_not_serialized_bytes(monkeypatch, tmp_path, tokens, should_compact):
    from src.config.config import global_config

    monkeypatch.setattr(global_config.chat, "context_prompt_token_high_watermark", 51200)
    host, _ = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    state = {"contextTokens": tokens, "visibleChars": 2_000_000, "archive": {"head": "head"}, "revision": 3}
    runner.bridge.request = AsyncMock(side_effect=[state, {"started": True, "jobId": "job"}, {"ready": True}, state])
    await runner._maintain_context()
    assert runner.bridge.request.await_count == (4 if should_compact else 1)
    if should_compact:
        started = runner.bridge.request.await_args_list[1]
        assert started.kwargs["preserveUnprocessed"] is True
        assert started.kwargs["summaryTaskName"] == "memory"
        assert started.kwargs["summaryTimeoutMs"] == 300000


@pytest.mark.asyncio
async def test_source_learning_is_independent_of_participation_and_sdk_compaction(monkeypatch, tmp_path):
    from types import MethodType
    from src.common.data_models.message_component_data_model import TextComponent
    from src.maisaka.context.messages import SessionBackedMessage
    from src.maisaka.runtime import MaisakaHeartFlowChatting

    source = original_source_message()
    source.raw_message = MessageSequence([TextComponent("real learning source")])
    message = SessionBackedMessage.from_session_message(
        source, raw_message=source.raw_message, visible_text="real learning source"
    )
    with wire_server() as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        host._runtime._joint_learning = SimpleNamespace(collect=AsyncMock(return_value=[]))
        host._runtime._trigger_pi_source_learning = MethodType(
            MaisakaHeartFlowChatting._trigger_pi_source_learning, host._runtime
        )
        host._runtime._chat_history.extend([message, history("not a learning source", "reference-like")])
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.ingest()
            await runner.ingest()
            assert host._runtime._joint_learning.collect.await_count == 2
            host._runtime._joint_learning.collect.assert_awaited_with([message])
            # The durable journal owns deduplication, including restart replay.
            assert requests == []  # Neither Luna decision nor main model/compaction is needed.
            assert await runner.pending_source_ids() == []
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_background_child_allows_new_parent_turn_and_delivers_completion_once(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    release = asyncio.Event()

    def route(request, requests):
        is_child = "# 独立任务" in json.dumps(request["messages"], ensure_ascii=False)
        outputs = [m for m in request["messages"] if m["role"] == "tool"]
        if is_child:
            return {"text": "CHILD_FINISHED"} if outputs else ("web_fetch", {})
        return (
            ("reply", {"text": "PARENT_CONTINUES"})
            if outputs
            else ("subagent", {"task": "BACKGROUND_TASK", "wait": False})
        )

    with wire_server([route] * 12) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, effects = engine()

        async def invoke(invocation, context, **kwargs):
            effects.append(invocation)
            if invocation.tool_name == "reply":
                return ToolExecutionResult("reply", True, content="parent sent", metadata={"pause_execution": True})
            await release.wait()
            return ToolExecutionResult("web_fetch", True, content="CHILD_PRIVATE_SOURCE")

        host._runtime._tool_registry.invoke = invoke
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            assert not release.is_set()
            before = next(
                r["messages"]
                for r in reversed(requests)
                if "PARENT_CONTINUES" not in json.dumps(r) and "# 独立任务" not in json.dumps(r, ensure_ascii=False)
            )
            host._runtime._chat_history.append(history("PARENT_NEW_CHAT", "new-chat"))
            await runner.run([])
            assert any("PARENT_NEW_CHAT" in json.dumps(r["messages"]) for r in requests)
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            handle = re.search(r'handle="(pi-child-[^\"]+)"', receipt.result["content"][0]["text"])[1]
            release.set()
            result = await runner.subagents.read(runner.session_id, handle, wait=True)
            assert result["conclusion"] == "CHILD_FINISHED"
            pending = await runner._db("pending_inputs", runner.session_id)
            assert len([p for p in pending if p.payload.get("piChildCompletion")]) == 1
            host._runtime._queue_proactive_turn.assert_called_once()
            await runner.run([])
            assert not await runner._db("pending_inputs", runner.session_id)
            main_requests = [r for r in requests if "# 独立任务" not in json.dumps(r["messages"], ensure_ascii=False)]
            assert "CHILD_FINISHED" in json.dumps(main_requests[-1]["messages"])
            assert "CHILD_PRIVATE_SOURCE" not in json.dumps(main_requests[-1]["messages"])
            assert all(field not in json.dumps(main_requests[-1]["messages"]) for field in ("host_tool_receipt", "evidence_handle", "conclusion_truncated", "No separately structured"))
            child_requests = [r for r in requests if "# 独立任务" in json.dumps(r["messages"], ensure_ascii=False)]
            assert child_requests[0]["tools"] == main_requests[0]["tools"]
            assert "PARENT_NEW_CHAT" not in json.dumps(child_requests[0]["messages"])
            assert main_requests[-1]["messages"][: len(before)] == before
        finally:
            release.set()
            await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("control, arguments", [("stop", {}), ("wait", {"seconds": 0})])
async def test_child_controls_do_not_change_parent_scheduler(monkeypatch, tmp_path, control, arguments):
    with wire_server() as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, effects = engine()
        original_specs = await host._runtime._tool_registry.list_tools(None)
        spec = ToolSpec(name=control, description=control, provider_name="maisaka_builtin", provider_type="builtin")
        host._runtime._tool_registry.list_tools.return_value = [*original_specs, spec]
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.ensure_initialized()
            child = runner.session_id + "/child/controls"
            runner._children[child] = {"tools": list(runner._tools)}
            result = await runner._tool(
                {"sessionId": child, "name": control, "toolCallId": "child-control", "arguments": arguments}
            )
            assert result["terminate"] is (control == "stop")
            assert effects == [] and host._runtime._agent_state == "idle"
            assert runner.session_id not in runner._terminated_sessions
            assert not requests
        finally:
            runner._children.clear()
            await runner.close()


@pytest.mark.asyncio
async def test_completion_inbox_deduplicates_without_extra_wake_during_parent_response(tmp_path):
    host, _ = engine()
    runner = PiGroupRunner(host, data_dir=tmp_path)
    try:
        await runner._ensure_storage()
        runner._model_running = True
        result = {"handle": "pi-child-once", "status": "completed", "conclusion": "done", "sources": []}
        await runner._deliver_child_completion(result)
        await runner._deliver_child_completion(result)
        pending = await runner._db("pending_inputs", runner.session_id)
        assert len(pending) == 1 and pending[0].payload["piChildCompletion"] is True
        host._runtime._queue_proactive_turn.assert_not_called()
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_selected_turn_before_reference_failure_wakes_on_restart(monkeypatch, tmp_path):
    references = AsyncMock(side_effect=RuntimeError("reference failure"))
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", references)
    with wire_server([("reply", {"text": "recovered answer"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        host._runtime._chat_history.append(history("unanswered question", "question"))
        first = PiGroupRunner(host, data_dir=tmp_path)
        try:
            with pytest.raises(RuntimeError, match="reference failure"):
                await first.run([])
            assert await first._db("pending_turn", first.session_id)
            assert not await first._db("pending_inputs", first.session_id)
            assert not requests
        finally:
            await first.close()
        references.side_effect = None
        references.return_value = []
        restored_host, _ = engine()
        restored_host._runtime._tool_registry.invoke = AsyncMock(
            return_value=ToolExecutionResult(
                "reply",
                True,
                content="sent",
                metadata={"pause_execution": True},
            )
        )
        restored = PiGroupRunner(restored_host, data_dir=tmp_path)
        try:
            await restored.ensure_initialized()
            restored_host._runtime._queue_proactive_turn.assert_not_called()
            await restored.restore_pending_turn()
            restored_host._runtime._queue_proactive_turn.assert_called_once()
            await restored.run([])
            assert len(requests) == 1
            assert "unanswered question" in json.dumps(requests[0]["messages"])
            assert await restored._db("pending_turn", restored.session_id) is None
        finally:
            await restored.close()


@pytest.mark.asyncio
async def test_native_completion_survives_lost_python_turn_ack(monkeypatch, tmp_path):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([("reply", {"text": "already sent"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        host._runtime._tool_registry.invoke = AsyncMock(
            return_value=ToolExecutionResult(
                "reply",
                True,
                content="sent receipt",
                metadata={"pause_execution": True},
            )
        )
        first = PiGroupRunner(host, data_dir=tmp_path)
        original = first._db

        async def lose_ack(method, *args, **kwargs):
            if method == "complete_turn":
                raise OSError("lost Python completion ack")
            return await original(method, *args, **kwargs)

        monkeypatch.setattr(first, "_db", lose_ack)
        try:
            with pytest.raises(OSError, match="completion ack"):
                await first.run([])
        finally:
            await first.close()
        new_host, _ = engine()
        restored = PiGroupRunner(new_host, data_dir=tmp_path)
        try:
            await restored.ensure_initialized()
            assert await restored._db("pending_turn", restored.session_id) is None
            new_host._runtime._queue_proactive_turn.assert_not_called()
            assert len(requests) == 1
            host._runtime._tool_registry.invoke.assert_awaited_once()
        finally:
            await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["gpt_web_search", "gpt_read_search_result"])
async def test_local_search_timeout_is_failed_receipt_and_allows_reply(monkeypatch, tmp_path, name):
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([(name, {}), ("reply", {"text": "search timed out"})]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        host._runtime._tool_registry.list_tools.return_value.append(
            ToolSpec(
                name=name,
                description=name,
                parameters_schema={"type": "object"},
                provider_name="local.gpt-web-search",
                provider_type="plugin",
                metadata={"plugin_id": "local.gpt-web-search", "registry_original_name": name},
            )
        )

        async def invoke(invocation, context, **kwargs):
            invoked.append(invocation)
            return ToolExecutionResult(
                invocation.tool_name,
                invocation.tool_name == "reply",
                content="Search timed out" if invocation.tool_name == name else "reply sent",
                metadata={"plugin_id": "local.gpt-web-search"} if invocation.tool_name == name else {},
            )

        host._runtime._tool_registry.invoke = invoke
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            assert receipt.status == "failed"
            assert receipt.result["isError"] is True
            await runner._db("assert_no_unresolved_executions", runner.session_id)
            assert [item.tool_name for item in invoked] == [name, "reply"]
        finally:
            await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("not_executed", [True, False])
@pytest.mark.parametrize("name", ["remember_memory", "reply"])
async def test_builtin_failed_results_allow_self_correction_with_or_without_metadata(monkeypatch, tmp_path, not_executed, name):
    from src.maisaka.builtin_tool.remember_memory import get_tool_spec as memory_spec
    from src.maisaka.builtin_tool.reply import get_tool_spec as reply_spec
    from src.core.tooling import ToolAvailabilityContext

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([
        (name, {"content": "note", "evidence_message_ids": ["m1"], "subject_name": "unknown"} if name == "remember_memory"
         else {"msg_id": "m1", "text": "command", "attach_at": ["missing-at"]}),
        ("reply", {"msg_id": "m1", "text": "need a valid target"}),
    ]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        specs = host._runtime._tool_registry.list_tools.return_value
        specs[:] = [spec for spec in specs if spec.name != name]
        specs.append(memory_spec() if name == "remember_memory" else reply_spec(
            ToolAvailabilityContext(is_group_chat=True)
        ))

        async def invoke(invocation, context, **kwargs):
            invoked.append(invocation)
            if invocation.call_id == "call-1":
                return ToolExecutionResult(name, False, error_message="target not found",
                    metadata={"execution_outcome": "not_executed"} if not_executed else {})
            return ToolExecutionResult("reply", True, content="reply sent")

        host._runtime._tool_registry.invoke = invoke
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            assert receipt.status == "failed"
            assert receipt.result["isError"] is True
            assert receipt.result["content"][0]["text"] == "target not found"
            assert [item.tool_name for item in invoked] == [name, "reply"]
            assert len(requests) >= 2
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_child_browser_navigation_timeout_returns_error_and_keeps_research_running(monkeypatch, tmp_path):
    from src.core.tooling import ToolAnnotation

    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server([
        ("subagent", {"task": "Verify match schedule"}),
        ("browser_navigate", {"url": "https://lolesports.com/schedule"}),
        {"text": "Browser timed out; previously retrieved official evidence remains available."},
        None,
    ]) as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, _ = engine()
        host._runtime._tool_registry.list_tools.return_value.append(ToolSpec(
            name="browser_navigate", description="Navigate to a URL",
            parameters_schema={"type": "object", "properties": {"url": {"type": "string"}}},
            provider_name="mcp", provider_type="mcp",
            annotation=ToolAnnotation(read_only=False, destructive=True),
            metadata={"server_name": "playwright", "registry_original_name": "browser_navigate"},
        ))

        async def invoke(invocation, context, **kwargs):
            return ToolExecutionResult(invocation.tool_name, False, error_message=
                "MCP browser_navigate timed out after 120 seconds")

        host._runtime._tool_registry.invoke = invoke
        runner = PiGroupRunner(host, data_dir=tmp_path)
        try:
            await runner.run([])
            parent_receipt = await runner._db("lookup_tool_execution", runner.session_id, "call-1")
            visible = parent_receipt.result["content"][0]["text"]
            handle = re.search(r'handle="(pi-child-[^\"]+)"', visible)[1]
            child = await asyncio.to_thread(runner.subagents._summary, runner.session_id, handle)
            assert child["status"] == "completed", child
            assert "previously retrieved official evidence" in child["conclusion"]
            assert len(requests) == 4
            with runner.subagents._connect() as connection:
                child_id = connection.execute("SELECT child FROM pi_children WHERE handle=?", (handle,)).fetchone()[0]
            receipt = await runner._db("lookup_tool_execution", child_id, "call-2")
            assert receipt.status == "failed"
            assert receipt.result["isError"] is True
            assert "120 seconds" in receipt.result["content"][0]["text"]
            assert "120 seconds" in json.dumps(requests[2]["messages"])
        finally:
            await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("name,server,url", [
    ("browser_evaluate", "playwright", "https://example.org"),
    ("browser_navigate", "other-server", "https://example.org"),
    ("browser_navigate", "playwright", "javascript:submitForm()"),
])
async def test_other_provider_failed_results_return_to_child_without_pausing(tmp_path, name, server, url):
    from src.core.tooling import ToolAnnotation
    from src.maisaka.pi_runtime.runner import pi_tool

    host, _ = engine()
    spec = ToolSpec(
        name=name, description=name, parameters_schema={"type": "object"},
        provider_name="mcp", provider_type="mcp",
        annotation=ToolAnnotation(read_only=False, destructive=True),
        metadata={"server_name": server, "registry_original_name": name},
    )
    host._runtime._tool_registry.list_tools.return_value = [spec]
    host._runtime._tool_registry.invoke = AsyncMock(return_value=ToolExecutionResult(name, False, error_message="timeout"))
    runner = PiGroupRunner(host, data_dir=tmp_path)
    child = runner.session_id + "/child/failed-action"
    try:
        await runner._ensure_storage()
        runner._tools = {name: pi_tool(spec.to_llm_definition())}
        runner._bindings = {name: spec.execution_contract()}
        runner._children[child] = {"tools": [name]}
        result = await runner._tool({"sessionId": child, "name": name, "toolCallId": "action-1", "arguments": {"url": url}})
        assert result["isError"] is True
        receipt = await runner._db("lookup_tool_execution", child, "action-1")
        assert receipt.status == "failed"
    finally:
        runner._children.clear()
        await runner.close()
