"""Capacity admission at the final wire boundary, never an input truncation policy.

OpenAI chat/responses use an HTTPX request hook after SDK transforms and extra_body
replacement. Counters receive immutable bytes, not the mutable internal request.
No model-name heuristics, cache discounts or universal chars/4 estimates are used.
Gemini, external ACP and plugin RPC do not expose a verified final-wire contract
here: strict mode rejects them and audit mode explicitly reports the gap.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Iterator, Literal, Sequence, TypeVar
from urllib.parse import urlsplit

import asyncio
import hashlib
import json

from src.common.logger import get_logger
from src.config.model_configs import APIProvider, ModelInfo, RequestAdmissionConfig

logger = get_logger("request_admission")


@dataclass(frozen=True, slots=True)
class RequestRoute:
    provider: str
    endpoint: str
    model: str
    protocol: str


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """The actual final body; parsing returns a new object and cannot mutate it."""

    route: RequestRoute
    body: bytes
    transport_fingerprint: str = ""

    @property
    def payload_hash(self) -> str:
        return hashlib.sha256(self.body).hexdigest()

    def json_body(self) -> Dict[str, Any]:
        value = json.loads(self.body)
        if not isinstance(value, dict):
            raise ValueError("final request JSON must be an object")
        return value


@dataclass(frozen=True, slots=True)
class InputMeasurement:
    """A bound/estimate applies to all input, including tools/schema/opaque state.

    An injected counter is a trusted extension: it must count the complete final
    payload for this exact route, include cached input, and label estimates as
    estimates. Provider count APIs are not invoked by any built-in counter.
    """

    kind: Literal["exact", "upper_bound", "estimate", "unknown"]
    tokens: int | None = None
    source: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    status: Literal["admitted", "over_limit", "unknown"]
    reason: str
    route: RequestRoute
    payload_hash: str
    input_measurement: InputMeasurement
    output_tokens: int | None = None
    safety_margin_tokens: int = 0

    @property
    def input_upper(self) -> int | None:
        if (
            self.input_measurement.kind in ("exact", "upper_bound")
            and isinstance(self.input_measurement.source, str)
            and self.input_measurement.source.strip()
        ):
            return _nonnegative_int(self.input_measurement.tokens)
        return None


class AdmissionDenied(Exception):
    """Local, non-provider failure. Never retry, downgrade or penalize a route."""

    def __init__(self, decision: AdmissionDecision) -> None:
        self.decision = decision
        super().__init__(f"request admission {decision.status}: {decision.reason}")


InputCounter = Callable[[PreparedRequest], Awaitable[InputMeasurement]]
_INPUT_COUNTERS: Dict[str, InputCounter] = {}
_REQUIRE_VERIFIED_CAPACITY: ContextVar[bool] = ContextVar("require_verified_capacity", default=False)


@contextmanager
def require_verified_capacity() -> Iterator[None]:
    """Require fail-closed admission for this async request tree without changing shared model config.

    Pi runs use this scope; legacy callers keep their explicit model policy.
    Child provider tasks inherit the requirement, while unrelated groups do not.
    No caller can use this scope to relax an already strict route.
    """
    token = _REQUIRE_VERIFIED_CAPACITY.set(True)
    try:
        yield
    finally:
        _REQUIRE_VERIFIED_CAPACITY.reset(token)


def register_input_counter(name: str, counter: InputCounter) -> None:
    """Install a trusted, payload-dependent counter; intended for provider adapters."""
    if not name.strip() or name in {"unknown", "utf8_bytes_bound"}:
        raise ValueError("counter name must be nonempty and not a built-in name")
    _INPUT_COUNTERS[name] = counter


def unregister_input_counter(name: str) -> None:
    _INPUT_COUNTERS.pop(name, None)


_NONROUTING_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "connection",
        "content-type",
        "content-length",
        "host",
        "user-agent",
    }
)


def calculate_transport_fingerprint(
    route: RequestRoute,
    *,
    headers: Sequence[tuple[str, str]],
    query: Sequence[tuple[str, str]],
    auth_header: str = "authorization",
    auth_query_name: str = "",
) -> str:
    """Fingerprint the exact route and routing metadata, without exposing authentication.

    Use the final HTTP request's multi-item header/query lists. SDK transport
    headers and configured credential fields are excluded; organization/project
    and custom routing metadata are included. This digest is an identity, NOT
    proof of token limits: an operator must verify the route contract separately.
    Counters receive only this digest, never these raw metadata values.
    """
    excluded_headers = _NONROUTING_HEADERS | {"authorization", auth_header.lower()}
    routing_headers = sorted(
        (str(key).lower(), str(value))
        for key, value in headers
        if str(key).lower() not in excluded_headers and not str(key).lower().startswith("x-stainless-")
    )
    routing_query = sorted(
        (str(key), str(value)) for key, value in query if not auth_query_name or key != auth_query_name
    )
    value = {
        "route": {
            "provider": route.provider,
            "endpoint": route.endpoint,
            "model": route.model,
            "protocol": route.protocol,
        },
        "headers": routing_headers,
        "query": routing_query,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _plain_text_payload(value: Any) -> bool:
    """Byte bounds cannot measure image/audio/file tokens or hidden provider state.

    Unknown typed content is conservative too. Function tools/schema remain text
    and are counted as part of the entire JSON, not just messages[].content.
    """
    if isinstance(value, list):
        return all(_plain_text_payload(item) for item in value)
    if not isinstance(value, dict):
        return True
    forbidden = {
        "image_url",
        "image",
        "input_image",
        "input_audio",
        "audio",
        "file_id",
        "file_data",
        "file_url",
        "encrypted_content",
        "previous_response_id",
        "conversation",
        "inline_data",
        "inlineData",
        "fileData",
        "thought_signature",
        "thoughtSignature",
        "cache_control",
        "cached_content",
        "prompt",
        "continuation",
        "context_management",
        "container",
    }
    if forbidden.intersection(value):
        return False
    content_type = value.get("type")
    if isinstance(content_type, str) and content_type not in {
        "text",
        "input_text",
        "output_text",
        "message",
        "function",
        "function_call",
        "function_call_output",
        "json_schema",
        "json_object",
        "object",
        "array",
        "string",
        "number",
        "integer",
        "boolean",
        "null",
        "refusal",
    }:
        return False
    for key, item in value.items():
        # Function parameter and response JSON schemas are serialized text, not
        # protocol media/state. A property named image_url/prompt/file_id is not
        # itself an image or remote template reference.
        if key in {"parameters", "schema"} and ("name" in value or content_type == "json_schema"):
            continue
        if not _plain_text_payload(item):
            return False
    return True


def _nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _effective_output_limit(body: Dict[str, Any], protocol: str) -> int | None:
    if protocol == "openai_chat":
        # Both limits are ambiguous across compatible providers: do not guess.
        fields = [key for key in ("max_completion_tokens", "max_tokens") if key in body]
    elif protocol == "openai_responses":
        fields = [key for key in ("max_output_tokens",) if key in body]
    else:
        return None
    if len(fields) != 1:
        return None
    output = _nonnegative_int(body[fields[0]])
    return output if output else None


def _output_reserve(body: Dict[str, Any], config: RequestAdmissionConfig, protocol: str) -> int | None:
    output = _effective_output_limit(body, protocol)
    if output is None or config.output_semantics == "unknown":
        return None
    if config.output_semantics == "separate_reasoning":
        reasoning = config.reasoning_tokens_upper
        if config.reasoning_budget_path:
            value: Any = body
            for part in config.reasoning_budget_path.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            reasoning = _nonnegative_int(value)
        if reasoning is None:
            return None
        output += reasoning
    # n creates parallel completions; the window is per completion, not a sum.
    return output


def _text_only_wire(body: Dict[str, Any], protocol: str) -> bool:
    # Only reviewed envelope fields are supported by the byte contract. An
    # unknown provider extension may retrieve remote state or expand templates;
    # a small reference is not an upper bound on that hidden input.
    common = {
        "model",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "stream",
        "metadata",
        "store",
        "user",
        "service_tier",
        "parallel_tool_calls",
        "reasoning",
        "reasoning_effort",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "thinking",
    }
    protocol_fields = {
        "openai_chat": {
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "stream_options",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "n",
        },
        "openai_responses": {"input", "instructions", "max_output_tokens", "text", "include", "truncation"},
    }
    return set(body).issubset(common | protocol_fields.get(protocol, set())) and _plain_text_payload(body)


async def assess_request(
    prepared: PreparedRequest,
    config: RequestAdmissionConfig,
    *,
    expected_route: RequestRoute,
) -> AdmissionDecision:
    """Pure assessment of a sealed payload; no network or parameter adjustments."""
    unknown = InputMeasurement("unknown", reason="no authoritative input measurement")
    output: int | None = None

    def result(status: Literal["admitted", "over_limit", "unknown"], reason: str) -> AdmissionDecision:
        return AdmissionDecision(
            status,
            reason,
            prepared.route,
            prepared.payload_hash,
            measurement,
            output,
            config.safety_margin_tokens,
        )

    measurement = unknown
    if prepared.route != expected_route:
        return result("unknown", "actual route differs from the declared capability route")
    if not config.capability_source.strip():
        return result("unknown", "route capability source is not declared")
    try:
        body = await asyncio.to_thread(prepared.json_body)
    except (ValueError, UnicodeError):
        return result("unknown", "final wire body is not a JSON object")
    output = _output_reserve(body, config, prepared.route.protocol)
    if config.input_counter == "utf8_bytes_bound":
        if not await asyncio.to_thread(_text_only_wire, body, prepared.route.protocol):
            measurement = InputMeasurement(
                "unknown", reason="multimodal, opaque, server-managed or unrecognized provider input"
            )
        elif config.tokens_per_utf8_byte_upper is None or config.protocol_overhead_tokens is None:
            measurement = InputMeasurement("unknown", reason="byte-bound contract is incomplete")
        else:
            measurement = InputMeasurement(
                "upper_bound",
                len(prepared.body) * config.tokens_per_utf8_byte_upper + config.protocol_overhead_tokens,
                config.capability_source,
            )
    elif config.input_counter in _INPUT_COUNTERS:
        try:
            measurement = await _INPUT_COUNTERS[config.input_counter](prepared)
        except Exception as exc:
            measurement = InputMeasurement("unknown", reason=f"counter failed: {type(exc).__name__}")
    if not isinstance(measurement, InputMeasurement):
        measurement = InputMeasurement("unknown", reason="counter did not return InputMeasurement")
    upper = _nonnegative_int(measurement.tokens)
    if (
        measurement.kind not in ("exact", "upper_bound")
        or not isinstance(measurement.source, str)
        or not measurement.source.strip()
    ):
        upper = None
    margin = config.safety_margin_tokens
    # Check every independently provable overflow before missing measurements.
    # Unknown output does not excuse an already-overflowing input (or vice versa).
    output_floor = output if output is not None else _effective_output_limit(body, prepared.route.protocol)
    if output_floor is not None:
        if config.max_output_tokens is not None and output_floor > config.max_output_tokens:
            return result("over_limit", "effective output reserve exceeds the independent output limit")
        if config.context_window is not None and output_floor + margin > config.context_window:
            return result("over_limit", "effective output reserve plus margin exceeds the shared window")
    if upper is not None:
        if config.max_input_tokens is not None and upper + margin > config.max_input_tokens:
            return result("over_limit", "input upper bound plus margin exceeds the independent input limit")
        if config.context_window is not None and upper + margin > config.context_window:
            return result("over_limit", "input upper bound plus margin exceeds the shared window")
    if upper is None:
        return result("unknown", measurement.reason or "input has no trusted upper bound")
    if output is None:
        return result("unknown", "effective output limit or reasoning semantics is unknown")
    if config.context_window is None and (config.max_input_tokens is None or config.max_output_tokens is None):
        return result("unknown", "neither shared window nor both independent limits are declared")
    if config.context_window is not None and upper + output + margin > config.context_window:
        return result("over_limit", "input upper bound + effective output reserve + margin exceeds the shared window")
    return result("admitted", "all declared limits satisfied by a trusted input bound")


def admit_request(decision: AdmissionDecision, mode: str) -> None:
    if decision.status == "admitted":
        return
    if _REQUIRE_VERIFIED_CAPACITY.get() or mode == "strict" or decision.status == "over_limit":
        raise AdmissionDenied(decision)
    logger.warning(
        "request admission AUDIT ONLY (not capacity-safe): status=%s reason=%s provider=%s model=%s hash=%s",
        decision.status,
        decision.reason,
        decision.route.provider,
        decision.route.model,
        decision.payload_hash,
    )


def guard_unverified_route(
    *,
    mode: str,
    provider: str = "",
    endpoint: str = "",
    model: str = "",
    operation: str,
) -> None:
    """Guard a route (including external ACP) that has no final-wire contract."""
    route = RequestRoute(provider, endpoint, model, operation)
    decision = AdmissionDecision(
        "unknown",
        f"{operation} has no verified final-wire admission contract",
        route,
        "",
        InputMeasurement("unknown", reason="unsupported final-wire route"),
    )
    admit_request(decision, mode)


def guard_unsupported_route(model_info: ModelInfo, provider: APIProvider | None = None, *, operation: str) -> None:
    """Model-configured variant of the unverified external-route guard."""
    guard_unverified_route(
        mode=model_info.request_admission.mode,
        provider=provider.name if provider is not None else model_info.api_provider,
        endpoint=provider.base_url if provider is not None else "",
        model=model_info.model_identifier,
        operation=operation,
    )


@dataclass(slots=True)
class _AdmissionScope:
    config: RequestAdmissionConfig
    expected_route: RequestRoute
    denied: AdmissionDenied | None = None
    transport_unknown_reason: str = ""
    allowed_query: tuple[tuple[str, str], ...] = ()
    auth_header: str = "authorization"
    standard_route_headers: tuple[tuple[str, str], ...] = ()
    auth_header_override: bool = False
    trace_context: Any = None


_SCOPE: ContextVar[_AdmissionScope | None] = ContextVar("final_wire_admission", default=None)
_Result = TypeVar("_Result")


def admitted_response(
    protocol: str,
) -> Callable[[Callable[..., Awaitable[_Result]]], Callable[..., Awaitable[_Result]]]:
    """Bind immutable configuration to the final HTTP hook across SDK child tasks."""

    def decorate(method: Callable[..., Awaitable[_Result]]) -> Callable[..., Awaitable[_Result]]:
        @wraps(method)
        async def wrapped(self: Any, request: Any, *args: Any, **kwargs: Any) -> _Result:
            endpoint = "/responses" if protocol == "openai_responses" else "/chat/completions"
            scope = _AdmissionScope(
                request.model_info.request_admission.model_copy(deep=True),
                RequestRoute(
                    self.api_provider.name,
                    (
                        self.api_provider.base_url.rstrip("/")
                        if "://" in self.api_provider.base_url
                        else "http://" + self.api_provider.base_url.rstrip("/")
                    )
                    + endpoint,
                    request.model_info.model_identifier,
                    protocol,
                ),
            )
            # Custom routing metadata requires a separately verified fingerprint
            # and an exact counter. A proxy may select another model/template;
            # counters never receive credentials or raw transport metadata.
            if any(
                (
                    self.api_provider.default_headers,
                    self.api_provider.default_query,
                    request.extra_params.get("headers"),
                    request.extra_params.get("query"),
                )
            ):
                scope.transport_unknown_reason = "custom headers/query have no declared final-route contract"
            if self.api_provider.auth_type == "query":
                scope.allowed_query = ((self.api_provider.auth_query_name, self.api_provider.api_key),)
            scope.auth_header = self.api_provider.auth_header_name.lower()
            request_headers = request.extra_params.get("headers", {})
            if isinstance(request_headers, dict):
                scope.auth_header_override = bool(
                    {str(key).lower() for key in request_headers} & {"authorization", scope.auth_header}
                )
            scope.standard_route_headers = (
                ("openai-organization", self.api_provider.organization or ""),
                ("openai-project", self.api_provider.project or ""),
            )
            scope.trace_context = request.trace_context
            if scope.trace_context is not None:
                scope.trace_context.pending_admission = None
            token = _SCOPE.set(scope)
            try:
                return await method(self, request, *args, **kwargs)
            except Exception:
                # OpenAI wraps HTTPX hook failures as APIConnectionError. The
                # hook stores the original local exception before SDK wrapping;
                # restore it even if an adapter already wrapped the SDK error.
                if scope.denied is not None:
                    raise scope.denied from None
                raise
            finally:
                _SCOPE.reset(token)

        return wrapped

    return decorate


def reraise_admission_denial() -> None:
    """Unwrap SDK transport errors before snapshotting them as provider failures."""
    scope = _SCOPE.get()
    if scope is not None and scope.denied is not None:
        raise scope.denied from None


async def admit_http_request(request: Any) -> None:
    """Last HTTPX request hook. SDK bytes are assessed and sent unchanged.

    Headers/query values are deliberately not logged or sent to token counters.
    The path and final body identify the actual route; redirects are assessed
    again. OpenAI SDK retries are disabled by the adapter.
    """
    scope = _SCOPE.get()
    if scope is None:
        return  # embeddings, transcription and management are not LLM generation
    if scope.denied is not None:
        raise scope.denied  # no recount/retry of the identical rejected request
    if scope.trace_context is not None:
        scope.trace_context.pending_admission = AdmissionDecision(
            "unknown",
            "final-wire preparation did not finish before the task deadline",
            scope.expected_route,
            "",
            InputMeasurement("unknown", reason="preparation interrupted"),
        )
    body = bytes(await request.aread())
    endpoint = urlsplit(str(request.url))._replace(query="", fragment="").geturl()
    try:
        decoded = await asyncio.to_thread(json.loads, body)
        model = decoded.get("model", "") if isinstance(decoded, dict) else ""
        if not isinstance(model, str):
            model = ""
    except (ValueError, UnicodeError):
        model = ""
    route = RequestRoute(scope.expected_route.provider, endpoint, str(model), scope.expected_route.protocol)
    original_headers = tuple(request.headers.multi_items())
    original_query = tuple(request.url.params.multi_items())
    auth_query_name = scope.allowed_query[0][0] if scope.allowed_query else ""
    fingerprint = calculate_transport_fingerprint(
        route,
        headers=original_headers,
        query=original_query,
        auth_header=scope.auth_header,
        auth_query_name=auth_query_name,
    )
    prepared = PreparedRequest(route, body, fingerprint)
    if scope.trace_context is not None:
        scope.trace_context.pending_admission = AdmissionDecision(
            "unknown",
            "final-wire assessment did not finish before the task deadline",
            prepared.route,
            prepared.payload_hash,
            InputMeasurement("unknown", reason="assessment interrupted"),
        )
    standard_headers = {
        "accept",
        "accept-encoding",
        "connection",
        "content-type",
        "content-length",
        "host",
        "user-agent",
        "authorization",
        "openai-organization",
        "openai-project",
        scope.auth_header,
    }
    unknown_headers = any(key not in standard_headers and not key.startswith("x-stainless-") for key in request.headers)
    transport_unknown = scope.transport_unknown_reason
    if (
        original_query != scope.allowed_query
        or unknown_headers
        or any(request.headers.get(key, "") != value for key, value in scope.standard_route_headers)
    ):
        transport_unknown = "final transport routing metadata is not supported by the capability contract"
    expected_fingerprint = scope.config.transport_contract_fingerprint
    contract_verified = bool(
        expected_fingerprint
        and expected_fingerprint == fingerprint
        and scope.config.capability_source.strip()
        and scope.config.input_counter in _INPUT_COUNTERS
    )
    if expected_fingerprint and expected_fingerprint != fingerprint:
        transport_unknown = "final transport metadata differs from the declared route contract"
    # A configured credential is omitted from the digest, but a mismatched or
    # duplicated credential query cannot select an undeclared authenticated route.
    auth_query_mismatch = (
        bool(scope.allowed_query)
        and tuple(pair for pair in original_query if pair[0] == auth_query_name) != scope.allowed_query
    )
    if (transport_unknown and not contract_verified) or auth_query_mismatch or scope.auth_header_override:
        decision = AdmissionDecision(
            "unknown",
            (
                "request-level authentication override is outside the declared transport route contract"
                if scope.auth_header_override
                else "configured authentication query does not match"
                if auth_query_mismatch
                else transport_unknown
            ),
            prepared.route,
            prepared.payload_hash,
            InputMeasurement("unknown", reason="unverified transport route"),
        )
    else:
        decision = await assess_request(prepared, scope.config, expected_route=scope.expected_route)
        if transport_unknown and decision.status == "admitted" and decision.input_measurement.kind != "exact":
            decision = AdmissionDecision(
                "unknown",
                "custom transport contract requires an authoritative exact input counter",
                prepared.route,
                prepared.payload_hash,
                decision.input_measurement,
                decision.output_tokens,
                decision.safety_margin_tokens,
            )
    try:
        # A counter must not swallow cancellation then let a timed-out request
        # reach the network. Awaiting SDK child tasks must remain cancel-safe.
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise asyncio.CancelledError
        admit_request(decision, scope.config.mode)
        # Counters never receive request, only immutable PreparedRequest. Check
        # nonetheless to keep future async hooks from invalidating assessment.
        if (
            bytes(request.content) != prepared.body
            or str(request.url).split("?", 1)[0] != endpoint
            or tuple(request.headers.multi_items()) != original_headers
            or tuple(request.url.params.multi_items()) != original_query
        ):
            raise AdmissionDenied(
                AdmissionDecision(
                    "unknown",
                    "wire body changed after preparation",
                    prepared.route,
                    prepared.payload_hash,
                    decision.input_measurement,
                )
            )
    except AdmissionDenied as exc:
        scope.denied = exc
        raise
    else:
        if scope.trace_context is not None:
            scope.trace_context.pending_admission = None
