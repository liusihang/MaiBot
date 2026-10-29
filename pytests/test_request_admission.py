"""Offline final-wire admission, using the real OpenAI SDK and MockTransport."""

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock

import asyncio
import json

import httpx
import pytest

from src.config.model_configs import APIProvider, ModelInfo, RequestAdmissionConfig, TaskConfig
from src.llm_models import utils_model
from src.llm_models.model_client import openai_client
from src.llm_models.model_client.base_client import ResponseRequest
from src.llm_models.model_client.openai_responses_client import OpenAIResponsesClient
from src.llm_models.payload_content.context_item import ContextItemBuilder
from src.llm_models.request_admission import (
    AdmissionDenied,
    InputMeasurement,
    PreparedRequest,
    RequestRoute,
    admit_request,
    assess_request,
    calculate_transport_fingerprint,
    guard_unverified_route,
    register_input_counter,
    require_verified_capacity,
    unregister_input_counter,
)


def policy(**overrides):
    values = dict(
        mode="strict",
        capability_source="offline verified route contract",
        context_window=10000,
        safety_margin_tokens=7,
        output_semantics="includes_reasoning",
        input_counter="utf8_bytes_bound",
        tokens_per_utf8_byte_upper=1,
        protocol_overhead_tokens=3,
    )
    values.update(overrides)
    return RequestAdmissionConfig(**values)


def model(name="opaque-alias", **policy_overrides):
    return ModelInfo(
        model_type="llm",
        model_identifier=name,
        name=name,
        api_provider="offline",
        request_admission=policy(**policy_overrides),
    )


def request(model_info=None, *, text="hello", extra=None):
    return ResponseRequest(
        model_info=model_info or model(),
        context_items=[ContextItemBuilder().add_text_content(text).build()],
        max_tokens=100,
        extra_params=extra or {},
    )


def final_request(body, protocol="openai_chat"):
    endpoint = "https://offline.invalid/v1/" + ("responses" if protocol == "openai_responses" else "chat/completions")
    return PreparedRequest(
        RequestRoute("offline", endpoint, body.get("model", "opaque-alias"), protocol),
        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(),
    )


@pytest.fixture
def sdk_client(monkeypatch):
    """Every HTTP send goes only to an in-process mock, never a real endpoint."""
    sent = []

    def create(protocol="openai", handler=None):
        def respond(req):
            sent.append(req)
            if handler is not None:
                return handler(req)
            if protocol == "openai_responses":
                return httpx.Response(
                    200,
                    json={
                        "id": "resp-offline",
                        "object": "response",
                        "status": "completed",
                        "created_at": 1,
                        "model": "opaque-alias",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg-offline",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                            }
                        ],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "offline",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "opaque-alias",
                    "choices": [
                        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                    ],
                },
            )

        def http_client(**kwargs):
            kwargs.pop("verify", None)
            return httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)

        monkeypatch.setattr(openai_client, "DefaultAsyncHttpxClient", http_client)
        provider = APIProvider(
            name="offline",
            auth_type="none",
            base_url="https://offline.invalid/v1",
            client_type=protocol,
            max_retry=3,
        )
        cls = OpenAIResponsesClient if protocol == "openai_responses" else openai_client.OpenaiClient
        return cls(provider), sent

    return create


@pytest.mark.asyncio
async def test_declared_byte_bound_counts_entire_json_without_alias_heuristics():
    body = {"model": "totally-unknown-1m-alias", "max_tokens": 100, "messages": [{"role": "user", "content": "汉🙂"}]}
    prepared = final_request(body)
    decision = await assess_request(prepared, policy(), expected_route=prepared.route)
    assert decision.status == "admitted"
    assert decision.input_upper == len(prepared.body) + 3
    assert decision.output_tokens == 100
    with pytest.raises(FrozenInstanceError):
        prepared.body = b"changed"
    prepared.json_body()["messages"].clear()
    assert prepared.json_body()["messages"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
            ]
        },
        {"input": [{"type": "reasoning", "encrypted_content": "opaque"}]},
        {"previous_response_id": "remote-state"},
        {"tools": [{"type": "web_search"}]},
    ],
)
async def test_images_opaque_and_provider_managed_input_are_unknown(extra):
    prepared = final_request({"model": "opaque-alias", "max_tokens": 100, **extra})
    decision = await assess_request(prepared, policy(), expected_route=prepared.route)
    assert decision.status == "unknown"
    assert decision.input_upper is None
    with pytest.raises(AdmissionDenied):
        admit_request(decision, "strict")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"capability_source": ""},
        {"input_counter": "unknown"},
        {"tokens_per_utf8_byte_upper": None},
        {"protocol_overhead_tokens": None},
        {"context_window": None},
        {"output_semantics": "unknown"},
    ],
)
async def test_missing_capabilities_are_not_inferred(overrides):
    prepared = final_request({"model": "gpt-super-1000000", "max_tokens": 100, "messages": []})
    decision = await assess_request(prepared, policy(**overrides), expected_route=prepared.route)
    assert decision.status == "unknown"


@pytest.mark.asyncio
async def test_all_limits_and_actual_reasoning_reserve_are_checked():
    prepared = final_request({"model": "opaque-alias", "max_tokens": 100, "thinking": {"budget": 70}, "messages": []})
    measured = len(prepared.body) + 3
    config = policy(
        context_window=measured + 170 + 7,
        max_input_tokens=measured + 7,
        max_output_tokens=170,
        output_semantics="separate_reasoning",
        reasoning_tokens_upper=1,
        reasoning_budget_path="thinking.budget",
    )
    decision = await assess_request(prepared, config, expected_route=prepared.route)
    assert decision.status == "admitted"
    assert decision.output_tokens == 170
    for field, limit in [
        ("context_window", measured + 176),
        ("max_input_tokens", measured + 6),
        ("max_output_tokens", 169),
    ]:
        limited = config.model_copy(update={field: limit})
        denial = await assess_request(prepared, limited, expected_route=prepared.route)
        assert denial.status == "over_limit"
        with pytest.raises(AdmissionDenied):
            admit_request(denial, "audit")  # migration never permits known overflow


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "measurement",
    [
        InputMeasurement("estimate", 1, "test estimate"),
        InputMeasurement("exact", 1, ""),
        InputMeasurement("upper_bound", -1, "bad counter"),
        InputMeasurement("unknown"),
    ],
)
async def test_counter_cannot_promote_estimates_or_invalid_counts(measurement):
    async def counter(_):
        return measurement

    register_input_counter("test_counter", counter)
    try:
        prepared = final_request({"model": "opaque-alias", "max_tokens": 10})
        decision = await assess_request(prepared, policy(input_counter="test_counter"), expected_route=prepared.route)
        assert decision.status == "unknown"
    finally:
        unregister_input_counter("test_counter")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai", "openai_responses"])
async def test_final_sdk_bytes_tools_schema_and_output_override(sdk_client, protocol):
    client, sent = sdk_client(protocol)
    seen = []

    async def counter(prepared):
        seen.append(prepared)
        return InputMeasurement("upper_bound", len(prepared.body), "offline complete-wire counter")

    register_input_counter("wire_test", counter)
    extra = {
        "body": {
            "max_completion_tokens" if protocol == "openai" else "max_output_tokens": 321,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "inspect", "parameters": {"type": "object", "description": "schema" * 50}},
                }
            ]
            if protocol == "openai"
            else [
                {"type": "function", "name": "inspect", "parameters": {"type": "object", "description": "schema" * 50}}
            ],
            "response_format" if protocol == "openai" else "text": {
                "type": "json_schema",
                "description": "constraint" * 30,
            },
        },
    }
    req = request(model(input_counter="wire_test"), extra=extra)
    try:
        await client.get_response(req)
        assert len(seen) == len(sent) == 1
        assert seen[0].body == sent[0].content
        body = seen[0].json_body()
        assert body["max_completion_tokens" if protocol == "openai" else "max_output_tokens"] == 321
        assert "schema" * 50 in seen[0].body.decode()
        assert "constraint" * 30 in seen[0].body.decode()
        assert req.max_tokens == 100  # no silent output reduction
    finally:
        unregister_input_counter("wire_test")
        await client.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"max_completion_tokens": 99999},
        {"messages": [{"role": "user", "content": "after-hook" * 1000}]},
        {"tools": [{"type": "function", "function": {"name": "large", "description": "x" * 9999}}]},
        {"response_format": {"type": "json_schema", "schema": {"description": "x" * 9999}}},
    ],
)
async def test_extra_body_final_overflow_is_never_sent_or_retried(sdk_client, override):
    client, sent = sdk_client()
    try:
        with pytest.raises(AdmissionDenied) as caught:
            await client.get_response(request(model(context_window=500), extra={"body": override}))
        assert caught.value.decision.status == "over_limit"
        assert sent == []  # SDK event-hook errors unwrap instead of provider retry
        assert client.client.max_retries == 0
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_final_model_override_invalidates_original_capability(sdk_client):
    client, sent = sdk_client()
    try:
        with pytest.raises(AdmissionDenied) as caught:
            await client.get_response(request(extra={"body": {"model": "different-model"}}))
        assert caught.value.decision.status == "unknown"
        assert caught.value.decision.route.model == "different-model"
        assert sent == []
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_unknown_migration_audit_is_explicitly_not_a_guarantee(sdk_client):
    client, sent = sdk_client()
    try:
        req = request(model())
        req.model_info.request_admission = RequestAdmissionConfig()
        await client.get_response(req)
        assert len(sent) == 1
    finally:
        await client.client.close()


@pytest.mark.parametrize("route", ["gemini", "plugin_rpc", "acp_external", "unknown_plugin"])
def test_unsupported_routes_fail_closed_in_strict_mode(route):
    with pytest.raises(AdmissionDenied) as caught:
        guard_unverified_route(mode="strict", operation=route, model="external")
    assert caught.value.decision.status == "unknown"
    guard_unverified_route(mode="audit", operation=route, model="external")


def orchestrator(monkeypatch, client, models):
    task = TaskConfig(model_list=[item.name for item in models], max_tokens=100, selection_strategy="sequential")
    monkeypatch.setattr(utils_model.LLMOrchestrator, "_get_task_config_or_raise", lambda _: task)
    monkeypatch.setattr(
        utils_model.TempMethodsLLMUtils,
        "get_model_info_by_name",
        lambda name: next(m for m in models if m.name == name),
    )
    monkeypatch.setattr(utils_model.TempMethodsLLMUtils, "get_provider_by_name", lambda _: client.api_provider)
    monkeypatch.setattr(utils_model.client_registry, "get_client_class_instance_async", AsyncMock(return_value=client))
    monkeypatch.setattr(utils_model, "mark_request_succeeded", lambda *a, **k: None)
    return utils_model.LLMOrchestrator("planner", "maisaka.planner")


@pytest.mark.asyncio
async def test_normal_generate_denial_releases_reservation_without_penalty_or_fallback(monkeypatch, sdk_client):
    client, sent = sdk_client()
    engine = orchestrator(monkeypatch, client, [model("A", context_window=300), model("B")])
    try:
        with pytest.raises(AdmissionDenied):
            await engine.generate_response_async("hook-expanded prompt" * 500)
        assert sent == []
        assert engine.model_usage == {"A": (0, 0, 0), "B": (0, 0, 0)}
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_snapshot_and_oversized_summary_cannot_bypass_admission(monkeypatch, sdk_client):
    client, sent = sdk_client()
    info = model(context_window=300)
    engine = orchestrator(monkeypatch, client, [info])
    engine.request_type = "maisaka.context_compaction"
    try:
        with pytest.raises(AdmissionDenied):
            await engine.generate_response_from_snapshot_async(request(info, text="oversized summary input" * 500))
        assert sent == []
        assert engine.model_usage[info.name] == (0, 0, 0)
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_provider_fallback_reassesses_actual_route_without_budget_fallback(monkeypatch, sdk_client):
    client, sent = sdk_client(handler=lambda _: httpx.Response(503, json={"error": {"message": "offline unavailable"}}))
    client.api_provider.max_retry = 1
    engine = orchestrator(monkeypatch, client, [model("A"), model("B", context_window=200), model("C")])
    try:
        with pytest.raises(AdmissionDenied) as caught:
            await engine.generate_response_async("payload" * 50)
        assert caught.value.decision.route.model == "B"
        assert len(sent) == 1
        assert json.loads(sent[0].content)["model"] == "A"
        assert engine.model_usage == {"A": (0, 1, 0), "B": (0, 0, 0), "C": (0, 0, 0)}
    finally:
        await client.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"output_semantics": "unknown"},
        {"max_input_tokens": 10, "output_semantics": "unknown"},
    ],
)
async def test_known_input_overflow_is_not_hidden_by_unknown_output(overrides):
    prepared = final_request({"model": "opaque-alias", "messages": [{"role": "user", "content": "x" * 500}]})
    decision = await assess_request(prepared, policy(context_window=200, **overrides), expected_route=prepared.route)
    assert decision.status == "over_limit"
    with pytest.raises(AdmissionDenied):
        admit_request(decision, "audit")


@pytest.mark.asyncio
async def test_known_output_overflow_is_not_hidden_by_unknown_counter():
    prepared = final_request({"model": "opaque-alias", "max_tokens": 500})
    decision = await assess_request(
        prepared,
        policy(context_window=200, input_counter="unknown", output_semantics="unknown"),
        expected_route=prepared.route,
    )
    assert decision.status == "over_limit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"prompt": {"id": "pmpt_server_template", "variables": {"short": "reference"}}},
        {"unknown_provider_extension": {"id": "hidden-input"}},
        {"conversation": "conv_previous"},
    ],
)
async def test_server_template_and_unknown_envelope_never_get_byte_bound(sdk_client, extra):
    client, sent = sdk_client("openai_responses")
    try:
        with pytest.raises(AdmissionDenied) as caught:
            await client.get_response(request(extra={"body": extra}))
        assert caught.value.decision.status == "unknown"
        assert sent == []
    finally:
        await client.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"headers": {"X-Upstream-Model": "different"}},
        {"headers": {"OpenAI-Organization": "other-org"}},
        {"headers": {"Authorization": "Bearer mock-only"}},
        {"query": {"model": "different"}},
    ],
)
async def test_custom_transport_routing_is_unsupported_not_silently_ignored(sdk_client, extra):
    client, sent = sdk_client()
    try:
        with pytest.raises(AdmissionDenied) as caught:
            await client.get_response(request(extra=extra))
        assert caught.value.decision.status == "unknown"
        assert "transport" in caught.value.decision.reason or "headers/query" in caught.value.decision.reason
        assert "mock-only" not in str(caught.value)
        assert sent == []
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_cache_key_never_discounts_input():
    prepared = final_request(
        {"model": "opaque-alias", "max_tokens": 10, "prompt_cache_key": "fully-cached", "messages": []}
    )
    decision = await assess_request(prepared, policy(), expected_route=prepared.route)
    assert decision.input_upper == len(prepared.body) + 3


@pytest.mark.asyncio
async def test_request_mutation_during_counter_does_not_change_final_wire_or_policy(sdk_client):
    client, sent = sdk_client()
    req = request(model(input_counter="mutating_test"))
    captured = []

    async def counter(prepared):
        captured.append(prepared)
        req.extra_params["body"] = {"messages": [{"role": "user", "content": "late mutation" * 10000}]}
        req.model_info.request_admission.context_window = 1
        prepared.json_body()["messages"].clear()
        return InputMeasurement("upper_bound", len(prepared.body), "offline complete-wire contract")

    register_input_counter("mutating_test", counter)
    try:
        await client.get_response(req)
        assert len(sent) == 1
        assert sent[0].content == captured[0].body
        assert b"late mutation" not in sent[0].content
    finally:
        unregister_input_counter("mutating_test")
        await client.client.close()


@pytest.mark.asyncio
async def test_concurrent_requests_have_isolated_admission_scopes(sdk_client):
    import asyncio

    client, sent = sdk_client()
    try:
        results = await asyncio.gather(
            client.get_response(request(model("small", context_window=100), text="x" * 500)),
            client.get_response(request(model("large", context_window=10000), text="x" * 500)),
            return_exceptions=True,
        )
        assert isinstance(results[0], AdmissionDenied)
        assert not isinstance(results[1], BaseException)
        assert len(sent) == 1
        assert json.loads(sent[0].content)["model"] == "large"
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_streaming_path_uses_the_same_final_wire_gate(sdk_client):
    payload = {
        "id": "offline-stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "opaque-alias",
        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
    }
    client, sent = sdk_client(
        handler=lambda _: httpx.Response(
            200,
            text="data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )
    try:
        info = model()
        info.force_stream_mode = True
        await client.get_response(request(info))
        assert len(sent) == 1
        info.request_admission.context_window = 10
        with pytest.raises(AdmissionDenied):
            await client.get_response(request(info))
        assert len(sent) == 1
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_gemini_and_plugin_public_entries_fail_before_any_send():
    from types import SimpleNamespace

    from src.llm_models.model_client.gemini_client import GeminiClient
    from src.llm_models.model_client.plugin_client import PluginLLMClient

    provider = APIProvider(name="offline", client_type="gemini", api_key="offline-mock-only")
    gemini = object.__new__(GeminiClient)
    gemini.api_provider = provider
    gemini.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock())))
    supervisor = SimpleNamespace(invoke_llm_provider=AsyncMock())
    plugin = PluginLLMClient(provider, supervisor, "offline-plugin", "offline-plugin")
    for client in [gemini, plugin]:
        with pytest.raises(AdmissionDenied):
            await client.get_response(request())
    supervisor.invoke_llm_provider.assert_not_awaited()
    gemini.client.aio.models.generate_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_conversion_hook_expansion_is_counted(sdk_client, monkeypatch):
    client, sent = sdk_client()
    original = openai_client._convert_messages

    def hook_expanded(items):
        return [*original(items), {"role": "system", "content": "late-tool-schema-and-policy" * 500}]

    monkeypatch.setattr(openai_client, "_convert_messages", hook_expanded)
    try:
        with pytest.raises(AdmissionDenied) as caught:
            await client.get_response(request(model(context_window=500)))
        assert caught.value.decision.status == "over_limit"
        assert sent == []
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_compatibility_parameter_retry_rechecks_changed_final_body(sdk_client):
    client, sent = sdk_client(
        handler=lambda _: httpx.Response(
            400,
            json={
                "error": {"message": "Use 'max_completion_tokens' instead."},
            },
        )
    )
    seen = []

    async def counter(prepared):
        seen.append(prepared)
        # The offline counter fixture reports 100 for legacy, 400 for the
        # rewritten body. It proves the second dispatch cannot reuse approval.
        return InputMeasurement(
            "exact", 400 if "max_completion_tokens" in prepared.json_body() else 100, "offline counter"
        )

    register_input_counter("compat_retry_test", counter)
    cache_key = (client.api_provider.base_url, "compat-retry-offline")
    openai_client._MODELS_REQUIRING_MAX_COMPLETION_TOKENS.discard(cache_key)
    try:
        with pytest.raises(AdmissionDenied):
            await client.get_response(
                request(model("compat-retry-offline", context_window=300, input_counter="compat_retry_test"))
            )
        assert len(seen) == 2
        assert len(sent) == 1
        assert seen[0].json_body()["max_tokens"] == seen[1].json_body()["max_completion_tokens"] == 100
    finally:
        unregister_input_counter("compat_retry_test")
        openai_client._MODELS_REQUIRING_MAX_COMPLETION_TOKENS.discard(cache_key)
        await client.client.close()


@pytest.mark.asyncio
async def test_malformed_counter_result_still_reports_known_output_overflow():
    async def counter(_):
        return None

    register_input_counter("malformed_test", counter)
    try:
        prepared = final_request({"model": "opaque-alias", "max_tokens": 900})
        decision = await assess_request(
            prepared,
            policy(context_window=200, input_counter="malformed_test"),
            expected_route=prepared.route,
        )
        assert decision.status == "over_limit"
        assert isinstance(decision.input_measurement, InputMeasurement)
    finally:
        unregister_input_counter("malformed_test")


@pytest.mark.asyncio
@pytest.mark.parametrize("swallow_cancellation", [False, True])
async def test_counter_task_timeout_is_local_denial_not_provider_fallback(
    monkeypatch, sdk_client, swallow_cancellation
):
    import asyncio

    client, sent = sdk_client()
    engine = orchestrator(monkeypatch, client, [model("A", input_counter="timeout_test"), model("B")])
    engine.model_for_task.hard_timeout = 1
    calls = 0

    async def counter(_):
        nonlocal calls
        calls += 1
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            if not swallow_cancellation:
                raise
        return InputMeasurement("exact", 1, "offline late counter")

    register_input_counter("timeout_test", counter)
    try:
        with pytest.raises(AdmissionDenied) as caught:
            await engine.generate_response_async("small prompt")
        assert caught.value.decision.status == "unknown"
        assert calls == 1
        assert sent == []
        assert engine.model_usage == {"A": (0, 0, 0), "B": (0, 0, 0)}
    finally:
        unregister_input_counter("timeout_test")
        await client.client.close()


@pytest.mark.asyncio
async def test_schema_property_names_are_text_not_real_media_or_server_references(sdk_client):
    client, sent = sdk_client()
    schema = {
        "type": "object",
        "properties": {
            "image_url": {"type": "string"},
            "prompt": {"type": "string"},
            "file_id": {"type": "string"},
        },
    }
    req = request(
        extra={
            "body": {
                "tools": [{"type": "function", "function": {"name": "describe", "parameters": schema}}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}},
            }
        }
    )
    try:
        await client.get_response(req)
        assert len(sent) == 1
        assert b"image_url" in sent[0].content
        assert b"prompt" in sent[0].content
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_pi_strict_scope_blocks_unknown_audit_route_before_sdk_send(sdk_client):
    client, sent = sdk_client()
    req = request(model(mode="audit", capability_source=""))
    try:
        with require_verified_capacity():
            with pytest.raises(AdmissionDenied) as caught:
                await client.get_response(req)
        assert caught.value.decision.status == "unknown"
        assert sent == []
        assert req.model_info.request_admission.mode == "audit"
        # Scope restoration must not silently change legacy configuration.
        await client.get_response(req)
        assert len(sent) == 1
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_pi_strict_scope_inherits_children_but_not_unrelated_groups():
    ready = asyncio.Event()
    release = asyncio.Event()

    async def strict_group():
        with require_verified_capacity():
            ready.set()
            await release.wait()
            async def child():
                guard_unverified_route(mode="audit", operation="offline_child")
            with pytest.raises(AdmissionDenied):
                await asyncio.create_task(child())
        guard_unverified_route(mode="audit", operation="offline_restored")

    strict_task = asyncio.create_task(strict_group())
    await ready.wait()
    guard_unverified_route(mode="audit", operation="offline_unrelated_group")
    release.set()
    await strict_task


def test_transport_fingerprint_binds_route_and_routing_but_excludes_configured_credentials():
    route = final_request({"model": "opaque-alias"}).route
    def fingerprint(header_value="route-A", secret="secret-A", query_value="revision-A", target=route):
        return calculate_transport_fingerprint(
            target, headers=[("X-Route", header_value), ("Authorization", secret), ("Content-Length", "999")],
            query=[("api-key", secret), ("api-version", query_value)], auth_query_name="api-key",
        )
    baseline = fingerprint()
    assert fingerprint(secret="secret-B") == baseline
    assert fingerprint(header_value="route-B") != baseline
    assert fingerprint(query_value="revision-B") != baseline
    assert fingerprint(target=RequestRoute("other-provider", route.endpoint, route.model, route.protocol)) != baseline
    assert calculate_transport_fingerprint(route, headers=[], query=[("", "routing-value")]) != calculate_transport_fingerprint(
        route, headers=[], query=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,actual_header,allowed", [("exact", "route-A", True), ("exact", "route-B", False), ("upper_bound", "route-A", False)])
async def test_custom_transport_requires_exact_counter_and_matching_declared_contract(sdk_client, kind, actual_header, allowed):
    client, sent = sdk_client()
    route = final_request({"model": "opaque-alias"}).route
    fingerprint = calculate_transport_fingerprint(route, headers=[("X-Route", "route-A")], query=[])
    counted = []
    async def counter(prepared):
        counted.append(prepared)
        assert prepared.transport_fingerprint == fingerprint
        return InputMeasurement(kind, 1, "offline verified transport-aware counter")
    register_input_counter("transport_contract_test", counter)
    req = request(model(input_counter="transport_contract_test", transport_contract_fingerprint=fingerprint),
                  extra={"headers": {"X-Route": actual_header}})
    try:
        if allowed:
            await client.get_response(req)
            assert len(sent) == 1
        else:
            with pytest.raises(AdmissionDenied):
                await client.get_response(req)
            assert sent == []
        assert len(counted) == (0 if actual_header != "route-A" else 1)
    finally:
        unregister_input_counter("transport_contract_test")
        await client.client.close()


@pytest.mark.asyncio
async def test_declared_transport_contract_does_not_make_byte_count_authoritative(sdk_client):
    client, sent = sdk_client()
    route = final_request({"model": "opaque-alias"}).route
    fingerprint = calculate_transport_fingerprint(route, headers=[("X-Route", "route-A")], query=[])
    try:
        with pytest.raises(AdmissionDenied):
            await client.get_response(request(model(transport_contract_fingerprint=fingerprint),
                                              extra={"headers": {"X-Route": "route-A"}}))
        assert sent == []
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_transport_change_during_counter_does_not_reach_network(sdk_client):
    client, sent = sdk_client()
    route = final_request({"model": "opaque-alias"}).route
    fingerprint = calculate_transport_fingerprint(route, headers=[("X-Route", "route-A")], query=[])
    captured = []
    async def capture(req):
        captured.append(req)
    async def counter(prepared):
        captured[-1].headers["X-Route"] = "route-B"
        return InputMeasurement("exact", 1, "offline verified counter")
    client.client._client.event_hooks["request"].insert(0, capture)
    register_input_counter("transport_mutation_test", counter)
    try:
        with pytest.raises(AdmissionDenied, match="changed"):
            await client.get_response(request(model(input_counter="transport_mutation_test", transport_contract_fingerprint=fingerprint),
                                              extra={"headers": {"X-Route": "route-A"}}))
        assert sent == []
    finally:
        unregister_input_counter("transport_mutation_test")
        await client.client.close()


@pytest.mark.asyncio
async def test_transport_fingerprint_cannot_authorize_different_request_authentication(sdk_client):
    client, sent = sdk_client()
    route = final_request({"model": "opaque-alias"}).route
    fingerprint = calculate_transport_fingerprint(route, headers=[("Authorization", "not-the-provider-identity")], query=[])
    async def counter(_):
        raise AssertionError("An undeclared authentication identity must not reach the counter")
    register_input_counter("auth_override_contract_test", counter)
    try:
        with pytest.raises(AdmissionDenied, match="authentication override"):
            await client.get_response(request(model(input_counter="auth_override_contract_test", transport_contract_fingerprint=fingerprint),
                                              extra={"headers": {"Authorization": "not-the-provider-identity"}}))
        assert sent == []
    finally:
        unregister_input_counter("auth_override_contract_test")
        await client.client.close()
