"""Retained pure admission policy/serialization contracts plus retired-entry guards.

The Python SDK transport/orchestrator fixtures formerly here are deliberately
retired, not bypassed in tests. Their production responsibilities are covered by
runtime/pi_agent/test/provider.test.mjs (real official provider mock wire),
inference.test.mjs (real short AgentSession), harness-proof.test.mjs (native
prefix/persistence), and pytests/services/test_pi_harness_bridge.py (host entry
categories, snapshot, cancellation, failure/no retry, and two-loop recovery).

Old provider fallback, compatibility-parameter retry, async Python counters and
mandatory Pi strict overrides are NOT carried into the new architecture. Pi
uses explicit configured audit/strict semantics and never silently lowers the
output budget or retries a deterministic parameter error. Pure policy helpers
below remain only for legacy offline diagnostic/snapshot consumers.
"""

from dataclasses import FrozenInstanceError

import asyncio
import json

import pytest

from src.config.model_configs import APIProvider, ModelInfo, RequestAdmissionConfig
from src.llm_models.model_client import openai_client
from src.llm_models.model_client.base_client import ResponseRequest
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


@pytest.mark.parametrize("route", ["gemini", "plugin_rpc", "acp_external", "unknown_plugin"])
def test_unsupported_routes_fail_closed_in_strict_mode(route):
    with pytest.raises(AdmissionDenied) as caught:
        guard_unverified_route(mode="strict", operation=route, model="external")
    assert caught.value.decision.status == "unknown"
    guard_unverified_route(mode="audit", operation=route, model="external")


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
async def test_cache_key_never_discounts_input():
    prepared = final_request(
        {"model": "opaque-alias", "max_tokens": 10, "prompt_cache_key": "fully-cached", "messages": []}
    )
    decision = await assess_request(prepared, policy(), expected_route=prepared.route)
    assert decision.input_upper == len(prepared.body) + 3


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
async def test_legacy_opt_in_strict_scope_inherits_children_but_not_unrelated_groups():
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
            target,
            headers=[("X-Route", header_value), ("Authorization", secret), ("Content-Length", "999")],
            query=[("api-key", secret), ("api-version", query_value)],
            auth_query_name="api-key",
        )

    baseline = fingerprint()
    assert fingerprint(secret="secret-B") == baseline
    assert fingerprint(header_value="route-B") != baseline
    assert fingerprint(query_value="revision-B") != baseline
    assert fingerprint(target=RequestRoute("other-provider", route.endpoint, route.model, route.protocol)) != baseline
    assert calculate_transport_fingerprint(
        route, headers=[], query=[("", "routing-value")]
    ) != calculate_transport_fingerprint(
        route,
        headers=[],
        query=[],
    )


@pytest.mark.parametrize("client_type", ["openai", "openai_responses", "gemini", "plugin"])
def test_retired_python_provider_constructors_cannot_reach_model_transport(monkeypatch, client_type):
    from types import SimpleNamespace
    from src.llm_models.model_client import gemini_client, openai_responses_client, plugin_client

    def forbidden(*args, **kwargs):
        raise AssertionError("retired Python model SDK must not be created")

    monkeypatch.setattr(openai_client, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(gemini_client.genai, "Client", forbidden)
    provider = APIProvider(
        name="offline",
        client_type=client_type,
        api_key="offline-fixture",
        base_url="https://offline.invalid/v1",
    )
    classes = {
        "openai": openai_client.OpenaiClient,
        "openai_responses": openai_responses_client.OpenAIResponsesClient,
        "gemini": gemini_client.GeminiClient,
        "plugin": plugin_client.PluginLLMClient,
    }
    args = (
        (provider, SimpleNamespace(invoke_llm_provider=forbidden), "plugin", "plugin")
        if client_type == "plugin"
        else (provider,)
    )
    with pytest.raises(RuntimeError, match="retired"):
        classes[client_type](*args)
