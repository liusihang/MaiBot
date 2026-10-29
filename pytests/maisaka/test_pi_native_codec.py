from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.llm_models.payload_content.context_item import (
    ContextItemMeta,
    ContextTextPart,
    ContextToolCall,
    FunctionCallItem,
    ProviderOpaqueItem,
    ProviderReplayFragment,
    ProviderScope,
    ReasoningItem,
    UserMessageItem,
    build_provider_endpoint_fingerprint,
)
from src.maisaka.pi_runtime.codec import decode_items, encode_items
from src.maisaka.pi_runtime.provider import NativeReplayScopeError, assert_replay_scope, context_from_pi


def native_fixture():
    scope = ProviderScope(
        1,
        "openai_responses",
        "route",
        build_provider_endpoint_fingerprint("openai_responses", "https://example.test/v1"),
        "real-model",
    )
    replay = ProviderReplayFragment.from_payload(scope, {"encrypted_content": "opaque", "type": "reasoning", "id": "r"})
    meta = ContextItemMeta.create(logical_turn_id="turn", item_id="native-1")
    return ReasoningItem(meta, replay=replay), scope


def test_codec_preserves_native_bytes_scope_and_id():
    native, _ = native_fixture()
    opaque = ProviderOpaqueItem(replace(native.meta, item_id="opaque"), "unknown-native", replay=native.replay)
    call = FunctionCallItem(
        replace(native.meta, item_id="call"),
        ContextToolCall("c", "read", b'{"x":1}', b'{"thought_signature":"opaque"}'),
    )
    original = [native, opaque, call]
    restored = decode_items(encode_items(original))
    assert restored == original
    assert restored[0].replay.payload_json == native.replay.payload_json
    assert restored[2].tool_call.extra_content_json == call.tool_call.extra_content_json


def test_codec_rejects_unknown_constructor_and_corruption():
    with pytest.raises(ValueError):
        decode_items([{"$type": "eval", "fields": {}}])
    native, _ = native_fixture()
    payload = encode_items([native])
    payload[0]["fields"]["replay"]["fields"]["payload_sha256"] = "invalid"
    with pytest.raises(ValueError, match="校验失败"):
        decode_items(payload)


def test_route_guard_exact_match_and_every_scope_component():
    native, scope = native_fixture()
    provider = SimpleNamespace(
        client_type=scope.client_type, name=scope.provider_name, base_url="https://example.test/v1"
    )
    client = SimpleNamespace(api_provider=provider)
    model = SimpleNamespace(model_identifier=scope.model_identifier)
    assert_replay_scope([native], client, model)
    for name, value in (("client_type", "openai"), ("name", "other"), ("base_url", "https://other.test")):
        changed = SimpleNamespace(**vars(provider))
        setattr(changed, name, value)
        with pytest.raises(NativeReplayScopeError):
            assert_replay_scope([native], SimpleNamespace(api_provider=changed), model)
    with pytest.raises(NativeReplayScopeError):
        assert_replay_scope([native], client, SimpleNamespace(model_identifier="other"))


def test_payload_prefix_stable_and_native_not_projected():
    native, _ = native_fixture()
    system = {"role": "system", "content": "Stable persona", "toolsAdded": []}
    user = UserMessageItem(ContextItemMeta.create(item_id="chat"), (ContextTextPart("hello"),))
    first = [system, {"role": "user", "native": {"items": encode_items([user])}}]
    prefix = context_from_pi(first)
    second = context_from_pi(
        first
        + [
            {"role": "assistant", "content": [], "native": {"items": encode_items([native])}},
            {"role": "system", "content": "", "toolsAdded": [{"name": "new"}]},
        ]
    )
    assert second[: len(prefix)] == prefix
    assert second[-1] == native
    with pytest.raises(ValueError, match="requires canonical"):
        context_from_pi([system, {"role": "assistant", "content": [{"type": "text", "text": "untrusted projection"}]}])


@pytest.mark.asyncio
async def test_provider_exact_second_payload_and_route_pin():
    from src.maisaka.pi_runtime.provider import PiProvider

    native, scope = native_fixture()
    client = SimpleNamespace(
        api_provider=SimpleNamespace(
            client_type=scope.client_type, name=scope.provider_name, base_url="https://example.test/v1"
        )
    )
    model = SimpleNamespace(model_identifier=scope.model_identifier)
    seen = []

    class LLM:
        async def generate_response_with_context(self, context_factory, options):
            seen.append(context_factory(client, model))
            return SimpleNamespace(
                output_items=(native,),
                response="",
                tool_calls=[],
                stop_reason="stop",
                model_name="route-model",
                prompt_tokens=4,
                completion_tokens=2,
                prompt_cache_hit_tokens=0,
                total_tokens=6,
            )

    gateway = PiProvider(SimpleNamespace(_get_llm_chat_client=lambda kind: LLM()))
    initial = [{"role": "system", "content": "stable"}, {"role": "user", "content": "question", "timestamp": 123}]
    response = await gateway({"messages": initial, "tools": []})
    assert response["message"]["native"]["route"]["model_identifier"] == "real-model"
    await gateway(
        {
            "messages": initial + [response["message"], {"role": "user", "content": "next", "timestamp": 124}],
            "tools": [],
        }
    )
    assert seen[1][: len(seen[0])] == seen[0]
    assert seen[1][len(seen[0])] == native
    model.model_identifier = "fallback-model"
    with pytest.raises(NativeReplayScopeError):
        await gateway({"messages": initial + [response["message"]], "tools": []})
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_unknown_stop_reason_is_not_assumed_complete():
    from src.maisaka.pi_runtime.provider import PiProvider

    class LLM:
        async def generate_response_with_context(self, context_factory, options):
            return SimpleNamespace(response="", tool_calls=[], stop_reason="unknown")

    gateway = PiProvider(SimpleNamespace(_get_llm_chat_client=lambda kind: LLM()))
    with pytest.raises(ValueError, match="terminal response"):
        await gateway({"messages": [{"role": "system", "content": "fixed"}], "tools": []})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,reason,input_upper,control",
    [
        ("over_limit", "input upper bound + effective output reserve + margin exceeds the shared window", 50000, True),
        ("unknown", "no verified counter", None, False),
        ("over_limit", "effective output reserve exceeds the independent output limit", 50000, False),
    ],
)
async def test_capacity_control_only_for_verified_reducible_input(status, reason, input_upper, control):
    from src.llm_models.request_admission import AdmissionDenied
    from src.maisaka.pi_runtime.provider import PiProvider

    decision = SimpleNamespace(status=status, reason=reason, input_upper=input_upper, payload_hash="body-hash")

    class LLM:
        async def generate_response_with_context(self, context_factory, options):
            raise AdmissionDenied(decision)

    gateway = PiProvider(SimpleNamespace(_get_llm_chat_client=lambda kind: LLM()))
    params = {"messages": [{"role": "system", "content": "fixed"}], "tools": []}
    if control:
        result = await gateway(params)
        assert result["message"]["hostControl"] == "capacity_admission"
        assert result["message"]["hostContinuationRequired"] is True
        assert result["message"]["native"] == {"items": []}
    else:
        with pytest.raises(AdmissionDenied):
            await gateway(params)
