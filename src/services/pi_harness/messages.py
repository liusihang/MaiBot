"""Portable business projections; provider-native assistant packets stay in Node."""

from typing import Any, Dict, List, Sequence

import uuid

from src.common.data_models.llm_service_data_models import LLMResponseResult
from src.llm_models.exceptions import ReqAbortException, RespParseException
from src.llm_models.payload_content.context_item import (
    AssistantMessageItem,
    ContextImagePart,
    ContextItem,
    ContextItemMeta,
    ContextRefusalPart,
    ContextTextPart,
    ContextToolCall,
    FunctionCallItem,
    FunctionCallOutputItem,
    ProviderOpaqueItem,
    ProviderReplayFragment,
    ProviderScope,
    ReasoningItem,
    ReasoningRepresentation,
    SystemMessageItem,
    UserMessageItem,
    get_item_replay,
)


HANDLE_CLIENT_TYPE = "pi_harness_handle"
PUBLIC_ONLY_PREFIX = "pi-public-only:"


def _parts(parts: Sequence[Any]) -> List[Dict[str, Any]]:
    result = []
    for part in parts:
        if isinstance(part, ContextTextPart):
            result.append({"type": "text", "text": part.text})
        elif isinstance(part, ContextImagePart):
            result.append(
                {"type": "image", "data": part.image_base64, "mimeType": "image/" + part.normalized_image_format}
            )
        elif isinstance(part, ContextRefusalPart):
            result.append({"type": "text", "text": part.refusal})
        else:
            raise ValueError("Unsupported business content part")
    return result


def items_to_messages(items: Sequence[ContextItem]) -> List[Dict[str, Any]]:
    """Send opaque references for prior assistants, never reconstructed signatures."""
    messages: List[Dict[str, Any]] = []
    handles = set()
    index = 0
    while index < len(items):
        item = items[index]
        index += 1
        if item.meta.item_id.startswith(PUBLIC_ONLY_PREFIX):
            raise ValueError("Single-shot Pi output cannot be replayed; generate with retain_replay=True")
        replay = get_item_replay(item)
        if replay is not None:
            if replay.scope.client_type != HANDLE_CLIENT_TYPE:
                raise ValueError(
                    "Legacy provider replay must be explicitly migrated; Python may not reconstruct native Pi history"
                )
            handle = replay.materialize()["responseHandle"]
            if handle not in handles:
                messages.append({"role": "assistantRef", "handle": handle})
                handles.add(handle)
            continue
        timestamp = int(item.meta.timestamp.timestamp() * 1000)
        if isinstance(item, FunctionCallItem):
            # MCP/plugin public tool examples have no provider signature. Keep
            # their semantic call/result pairs, explicitly separate from replay.
            calls = [item]
            while index < len(items) and isinstance(items[index], FunctionCallItem):
                if get_item_replay(items[index]) is not None:
                    break
                calls.append(items[index])
                index += 1
            if any(call.meta.item_id.startswith(PUBLIC_ONLY_PREFIX) for call in calls):
                raise ValueError("Single-shot Pi output cannot be replayed; generate with retain_replay=True")
            if any(call.tool_call.materialize_extra_content() for call in calls):
                raise ValueError("External tool history cannot contain provider-native extra content")
            names = {call.tool_call.call_id: call.tool_call.func_name for call in calls}
            if len(names) != len(calls):
                raise ValueError("External tool history has duplicate call IDs")
            results = []
            seen = set()
            while index < len(items) and isinstance(items[index], FunctionCallOutputItem):
                output = items[index]
                if get_item_replay(output) is not None:
                    raise ValueError("External tool results cannot contain provider-native replay")
                if output.call_id not in names or output.call_id in seen:
                    raise ValueError("External tool history has unmatched or duplicate results")
                if output.tool_name and output.tool_name != names[output.call_id]:
                    raise ValueError("External tool history has a mismatched tool name")
                seen.add(output.call_id)
                results.append(
                    {
                        "toolCallId": output.call_id,
                        "toolName": names[output.call_id],
                        "content": [{"type": "text", "text": output.output}],
                        "isError": not output.success,
                        "timestamp": int(output.meta.timestamp.timestamp() * 1000),
                    }
                )
                index += 1
            if seen != set(names):
                raise ValueError("External tool history requires a complete consecutive call/result exchange")
            messages.append(
                {
                    "role": "externalToolExchange",
                    "provenance": "caller_public_history",
                    "calls": [
                        {
                            "id": call.tool_call.call_id,
                            "name": call.tool_call.func_name,
                            "arguments": call.tool_call.materialize_args(),
                        }
                        for call in calls
                    ],
                    "results": results,
                }
            )
            continue
        if isinstance(item, SystemMessageItem):
            parts = _parts(item.parts)
            if any(part["type"] != "text" for part in parts):
                raise ValueError("System message must be textual")
            messages.append({"role": "system", "content": "\n".join(part["text"] for part in parts)})
        elif isinstance(item, UserMessageItem):
            messages.append({"role": "user", "content": _parts(item.parts), "timestamp": timestamp})
        elif isinstance(item, FunctionCallOutputItem):
            messages.append(
                {
                    "role": "toolResult",
                    "toolCallId": item.call_id,
                    "toolName": item.tool_name,
                    "content": [{"type": "text", "text": item.output}],
                    "isError": not item.success,
                    "timestamp": timestamp,
                }
            )
        elif isinstance(item, AssistantMessageItem):
            # Official Pi assistant messages cannot contain image examples.
            # Do not silently drop them or pretend they are native model output.
            if any(isinstance(part, ContextImagePart) for part in item.parts):
                raise ValueError("Pi assistant examples support plain text only; put example images in user messages")
            # Explicit caller-supplied portable example, not a provider output.
            messages.append({"role": "assistantExample", "content": _parts(item.parts), "timestamp": timestamp})
        else:
            raise ValueError("Assistant tool/reasoning/native state requires a live Node response handle")
    return messages


def validate_response_outcome(payload: Dict[str, Any]) -> None:
    """Reject failed or incomplete tool proposals before exposing business output."""
    message = payload["message"]
    stop = message.get("stopReason")
    if stop == "aborted":
        raise ReqAbortException("Pi model operation was aborted")
    if stop == "error":
        raise RespParseException(message="Pi model operation failed")
    if stop not in {"stop", "toolUse", "length"}:
        raise RespParseException(message="Pi returned an unknown completion outcome")
    if stop == "length" and any(block.get("type") == "toolCall" for block in message.get("content", [])):
        raise RespParseException(message="Pi returned a truncated tool proposal; tool calls cannot be executed")


def response_result(payload: Dict[str, Any], *, retain_replay: bool = True) -> LLMResponseResult:
    validate_response_outcome(payload)
    message, route = payload["message"], payload["route"]
    handle = payload.get("responseHandle")
    if retain_replay and (not isinstance(handle, str) or not handle):
        raise ValueError("Pi inference must return an opaque native response handle when replay is retained")
    if not retain_replay and handle is not None:
        raise ValueError("Pi returned an unexpected native handle for a single-shot operation")
    replay = None
    if retain_replay:
        scope = ProviderScope(
            1, HANDLE_CLIENT_TYPE, route["apiProvider"], "node-owned-handle", route["modelIdentifier"]
        )
        replay = ProviderReplayFragment.from_payload(scope, {"responseHandle": handle})
    turn = uuid.uuid4().hex
    items = []
    for block in message["content"]:
        meta = ContextItemMeta.create(
            logical_turn_id=turn,
            item_id=None if retain_replay else PUBLIC_ONLY_PREFIX + uuid.uuid4().hex,
        )
        if block["type"] == "text":
            if block.get("text"):
                items.append(AssistantMessageItem(meta, (ContextTextPart(block["text"]),), replay=replay))
        elif block["type"] == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                items.append(
                    ReasoningItem(
                        meta, text_parts=(thinking,), representation=ReasoningRepresentation.RAW_TEXT, replay=replay
                    )
                )
            else:
                items.append(ProviderOpaqueItem(meta, "pi_thinking", replay=replay))
        elif block["type"] == "toolCall":
            items.append(
                FunctionCallItem(
                    meta,
                    ContextToolCall.create(call_id=block["id"], func_name=block["name"], args=block["arguments"]),
                    replay=replay,
                )
            )
        else:
            raise ValueError("Unsupported Pi assistant business projection")
    if not items:
        items.append(
            ProviderOpaqueItem(
                ContextItemMeta.create(
                    logical_turn_id=turn,
                    item_id=None if retain_replay else PUBLIC_ONLY_PREFIX + uuid.uuid4().hex,
                ),
                "pi_empty_output",
                replay=replay,
            )
        )
    usage = payload.get("usage") or message.get("usage", {})
    return LLMResponseResult(
        output_items=tuple(items),
        model_name=route["modelName"],
        stop_reason=message["stopReason"],
        prompt_tokens=int(usage.get("input", 0)) + int(usage.get("cacheRead", 0)) + int(usage.get("cacheWrite", 0)),
        completion_tokens=int(usage.get("output", 0)),
        total_tokens=int(usage.get("totalTokens", 0)),
        prompt_cache_hit_tokens=int(usage.get("cacheRead", 0)),
        prompt_cache_miss_tokens=int(usage.get("input", 0)) + int(usage.get("cacheWrite", 0)),
        prompt_cache_reported=bool(usage.get("cacheReported", usage.get("cacheRead") or usage.get("cacheWrite"))),
        wire_protocol="pi_harness",
        provider_response={
            "route": route,
            "replayRetained": retain_replay,
            **({"responseHandle": handle} if handle else {}),
        },
    )
