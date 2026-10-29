"""Independent, offline acceptance checks for evidence/context repairs.

All model requests terminate at a fake client. These tests never create a bot
runtime, connect to a provider, or send platform messages.
"""

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from src.chat.message_receive.message import SessionMessage
from src.common.data_models.llm_service_data_models import LLMResponseResult
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.config.config import global_config
from src.config.model_configs import ModelInfo
from src.llm_models.model_client.base_client import ResponseRequest
from src.llm_models.payload_content.context_item import (
    AssistantMessageItem,
    ContextItemBuilder,
    ContextItemMeta,
    ContextTextPart,
    ContextToolCall,
    FunctionCallItem,
    FunctionCallOutputItem,
    ProviderOpaqueItem,
    ProviderReplayFragment,
    ProviderScope,
    ReasoningItem,
    RoleType,
    UserMessageItem,
    get_item_text,
)
from src.maisaka.chat_loop_service import MaisakaChatLoopService
from src.maisaka import reasoning_engine as engine_module
from src.maisaka.context import compaction
from src.maisaka.context.messages import (
    ContentContextMessage,
    ModelOutputContextMessage,
    SessionBackedMessage,
    ToolResultMediaMessage,
    ToolResultMessage,
)
from src.maisaka.context.post_processor import process_chat_history_after_cycle
from src.maisaka.reasoning_engine import MaisakaReasoningEngine


def _user(text: str) -> ContentContextMessage:
    return ContentContextMessage(MessageSequence([TextComponent(text)]), text, datetime.now())


def _native_turn(turn_id: str) -> list:
    scope = ProviderScope(1, "openai_responses", "offline-test", "offline-endpoint", "offline-model")
    reasoning = ReasoningItem(
        meta=ContextItemMeta.create(logical_turn_id=turn_id),
        replay=ProviderReplayFragment.from_payload(
            scope, {"id": turn_id + "-reason", "type": "reasoning", "encrypted_content": "TEST_ONLY_OPAQUE"}
        ),
    )
    call_id = turn_id + "-call"
    call = FunctionCallItem(
        meta=ContextItemMeta.create(logical_turn_id=turn_id),
        tool_call=ContextToolCall.create(call_id=call_id, func_name="query_memory", args={"query": turn_id}),
        replay=ProviderReplayFragment.from_payload(
            scope,
            {"type": "function_call", "call_id": call_id, "name": "query_memory", "arguments": "{}"},
        ),
    )
    opaque = ProviderOpaqueItem(
        meta=ContextItemMeta.create(logical_turn_id=turn_id),
        provider_type="future_provider_state",
        replay=ProviderReplayFragment.from_payload(scope, {"type": "future_provider_state", "id": turn_id}),
    )
    commentary = AssistantMessageItem(
        meta=ContextItemMeta.create(logical_turn_id=turn_id),
        parts=(ContextTextPart("A real, unchanged model message."),),
        phase="commentary",
        replay=ProviderReplayFragment.from_payload(
            scope, {"type": "message", "phase": "commentary", "id": turn_id + "-message"}
        ),
    )
    return [
        ModelOutputContextMessage(reasoning),
        ModelOutputContextMessage(commentary),
        ModelOutputContextMessage(opaque),
        ModelOutputContextMessage(call),
        ToolResultMessage("source excerpt " * 300, datetime.now(), call_id, turn_id, "query_memory"),
        ToolResultMediaMessage(
            MessageSequence([TextComponent("[tool media reference]")]),
            "[tool media reference]",
            datetime.now(),
            tool_call_id=call_id,
        ),
    ]


def _offline_service(monkeypatch: pytest.MonkeyPatch, *, hook, prompt_tokens: int = 100):
    requests = []

    class FakeClient:
        async def generate_response_with_context(self, context_factory, options):
            requests.append((context_factory(None), options))
            result = LLMResponseResult.from_portable_output(response="offline result", model_name="offline-model")
            result.prompt_tokens = prompt_tokens
            result.completion_tokens = 1
            result.total_tokens = prompt_tokens + 1
            return result

    class HookManager:
        async def invoke_hook(self, name: str, **kwargs: Any):
            return SimpleNamespace(kwargs=hook(name, kwargs))

    monkeypatch.setattr(global_config.debug, "save_maisaka_reasoning", False)
    monkeypatch.setattr(global_config.agent, "planner_enabled", False)
    service = MaisakaChatLoopService(chat_system_prompt="Offline acceptance system prompt.")
    monkeypatch.setattr(service, "_get_llm_chat_client", lambda _: FakeClient())
    monkeypatch.setattr(service, "_get_runtime_manager", lambda: HookManager())
    return service, requests


@pytest.mark.asyncio
@pytest.mark.parametrize("extension", ["message", "tool_schema"])
async def test_high_watermark_measures_final_post_hook_input(monkeypatch, extension):
    monkeypatch.setattr(global_config.chat, "context_visible_char_high_watermark", 10000)
    monkeypatch.setattr(global_config.chat, "context_prompt_token_high_watermark", 1000)
    large_text = "hook extension " * 1000

    def hook(name, kwargs):
        if name == "maisaka.planner.before_request":
            if extension == "message":
                kwargs["items"][0]["parts"][0]["text"] += large_text
            else:
                kwargs["tool_definitions"] = [{
                    "type": "function",
                    "function": {
                        "name": "offline_tool",
                        "description": large_text,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }]
        return kwargs

    service, requests = _offline_service(monkeypatch, hook=hook)
    await service.chat_loop_step([], tool_definitions=[])

    assert len(requests) == 1
    assert service.last_request_visible_chars >= len(large_text)
    assert service.last_reported_prompt_tokens == 100
    assert service.context_budget_exceeded is True


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_tokens", "hook_tokens", "exceeded"), [(2000, 0, True), (100, 999999, False)])
async def test_budget_uses_provider_usage_not_after_response_hook(monkeypatch, provider_tokens, hook_tokens, exceeded):
    monkeypatch.setattr(global_config.chat, "context_visible_char_high_watermark", 120000)
    monkeypatch.setattr(global_config.chat, "context_prompt_token_high_watermark", 1000)

    def hook(name, kwargs):
        if name == "maisaka.planner.after_response":
            kwargs["prompt_tokens"] = hook_tokens
        return kwargs

    service, _ = _offline_service(monkeypatch, hook=hook, prompt_tokens=provider_tokens)
    response = await service.chat_loop_step([], tool_definitions=[])

    assert response.prompt_tokens == hook_tokens  # The public hook contract is unchanged.
    assert service.last_reported_prompt_tokens == provider_tokens
    assert service.context_budget_exceeded is exceeded


@pytest.mark.asyncio
async def test_archive_and_system_hook_preserve_active_native_items(monkeypatch):
    old, active = _native_turn("completed"), _native_turn("active")
    first, next_question = _user("old question"), _user("new question")
    history = [first, *old, next_question, *active]
    planned = process_chat_history_after_cycle(
        history, max_context_size=40, force_compaction=True,
        active_logical_turn_id="active", consume_references=False,
    )
    assert planned.removed_messages == old
    assert planned.history == [first, next_question, *active]
    assert all(message in history for message in old)  # Planning is not a commit.

    def hook(name, kwargs):
        if name == "maisaka.planner.before_request":
            kwargs["items"][0]["parts"][0]["text"] += "\nTrusted runtime clock."
        return kwargs

    service, requests = _offline_service(monkeypatch, hook=hook)
    await service.chat_loop_step(planned.history, tool_definitions=[], logical_turn_id="active")
    sent = requests[0][0]
    active_model_items = [message.output_item for message in active if isinstance(message, ModelOutputContextMessage)]
    retained_ids = [item.meta.item_id for item in sent]
    assert [item.meta.item_id for item in active_model_items] == [
        item_id for item_id in retained_ids if item_id in {item.meta.item_id for item in active_model_items}
    ]
    for original in active_model_items:
        forwarded = next(item for item in sent if item.meta.item_id == original.meta.item_id)
        assert forwarded is original
        assert forwarded.replay.payload_sha256 == original.replay.payload_sha256
    assert next(item for item in sent if isinstance(item, AssistantMessageItem)).phase == "commentary"
    assert [item.tool_call.call_id for item in sent if isinstance(item, FunctionCallItem)] == ["active-call"]
    assert [item.call_id for item in sent if isinstance(item, FunctionCallOutputItem)] == ["active-call"]


@pytest.mark.parametrize("snapshot_covers_tools", [True, False])
def test_archive_summary_preserves_snapshot_and_encodes_uncovered_tools_as_data(monkeypatch, snapshot_covers_tools):
    first, next_question = _user("old question"), _user("new question")
    old, active = _native_turn("completed"), _native_turn("active")
    history = [first, *old, next_question, *active]
    planned = process_chat_history_after_cycle(
        history, max_context_size=40, force_compaction=True,
        active_logical_turn_id="active", consume_references=False,
    )
    covered = history if snapshot_covers_tools else [first]
    prefix = [ContextItemBuilder().set_role(RoleType.System).add_text_content("summary test").build()]
    prefix.extend(message.to_context_item() for message in covered)
    snapshot = compaction.PlannerRequestSnapshot(
        ResponseRequest(
            model_info=ModelInfo(model_type="llm", name="offline", model_identifier="offline", api_provider="offline"),
            context_items=list(prefix),
        ),
        tuple(covered),
    )
    monkeypatch.setattr(compaction, "load_prompt", lambda *args, **kwargs: "Return only an offline state summary.")

    request = compaction._build_compaction_request(
        planned.removed_messages,
        planner_snapshot=snapshot,
        previous_summary=None,
        recent_messages=planned.history,
        max_summary_chars=6000,
    )

    assert request.context_items[:-1] == prefix
    assert snapshot.request.context_items == prefix
    assert len(snapshot.request.context_items) + 1 == len(request.context_items)
    tail = request.context_items[-1]
    assert isinstance(tail, UserMessageItem)
    tail_text = get_item_text(tail)
    assert "TEST_ONLY_OPAQUE" not in tail_text  # Never turn private native state into plain summary material.
    if snapshot_covers_tools:
        assert '"kind": "tool_call"' not in tail_text
    else:
        assert '"kind": "tool_call"' in tail_text
        assert '"kind": "tool_result"' in tail_text
        assert '"call_id": "completed-call"' in tail_text
        assert not any(isinstance(item, FunctionCallItem) for item in request.context_items)


@pytest.mark.asyncio
@pytest.mark.parametrize("removal_kind", ["completed_tool_archive", "ordinary_chat_trim"])
async def test_luna_nonparticipation_cannot_commit_unsummarized_history_loss(monkeypatch, removal_kind):
    """No main-model call is allowed here, but no summary means no archive commit."""
    if removal_kind == "completed_tool_archive":
        history = [_user("first"), *_native_turn("old"), _user("second"), *_native_turn("new")]
        max_context_size, exceeded = 40, True
    else:
        history = []
        for index in range(9):
            source = SessionMessage(f"message-{index}", datetime.now(), "offline")
            source.session_id = "offline-session"
            source.message_info = MessageInfo(UserInfo("user-a", "A"))
            source.processed_plain_text = f"message {index}"
            source.raw_message = MessageSequence([TextComponent(source.processed_plain_text)])
            history.append(SessionBackedMessage.from_session_message(
                source, raw_message=source.raw_message, visible_text=source.processed_plain_text,
            ))
        max_context_size, exceeded = 5, False
    planned = process_chat_history_after_cycle(
        history, max_context_size=max_context_size, force_compaction=exceeded, consume_references=False,
    )
    assert planned.removed_messages  # Both branches must actually propose substantive loss.
    original_messages = list(history)
    snapshot = object()
    runtime = SimpleNamespace(
        session_id="offline-session",
        log_prefix="[offline]",
        _chat_history=history,
        _max_context_size=max_context_size,
        _get_pending_wait_tool_call_ids=lambda: set(),
        _chat_loop_service=SimpleNamespace(planner_request_snapshot=snapshot, context_budget_exceeded=exceeded),
        _monitor_run_id="offline-run",
        _log_history_trimmed=Mock(),
        _trigger_trimmed_history_learning=AsyncMock(),
    )
    engine = MaisakaReasoningEngine(runtime)
    summarize = AsyncMock()
    monkeypatch.setattr(engine_module, "build_context_summary", summarize)
    monkeypatch.setattr(engine_module, "emit_flow_step", AsyncMock())

    await engine._post_process_chat_history_after_cycle(
        SimpleNamespace(cycle_id=1, time_records={}),
        enable_compaction=False,
        consume_references=False,
    )

    summarize.assert_not_awaited()
    assert runtime._chat_history == original_messages
    assert all(current is original for current, original in zip(runtime._chat_history, original_messages, strict=True))
    assert runtime._chat_loop_service.planner_request_snapshot is snapshot
    assert runtime._chat_loop_service.context_budget_exceeded is exceeded
    runtime._trigger_trimmed_history_learning.assert_not_awaited()
