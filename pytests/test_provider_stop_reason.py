"""Provider terminal signals survive the Python/Pi boundary; no network or tools execute."""

from types import SimpleNamespace

import pytest
from google.genai.types import Candidate, Content, FunctionCall, GenerateContentResponse, Part
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from src.common.data_models.llm_service_data_models import LLMResponseResult
from src.config.model_configs import ReasoningParseMode, ToolArgumentParseMode
from src.llm_models.exceptions import RespParseException
from src.llm_models.model_client import gemini_client, openai_client, openai_responses_client
from src.llm_models.model_client.base_client import APIResponse
from src.llm_models.utils_model import LLMOrchestrator


OPENAI_OPTIONS = {
    "reasoning_parse_mode": ReasoningParseMode.NONE,
    "tool_argument_parse_mode": ToolArgumentParseMode.REPAIR,
    "reasoning_key": "reasoning_content",
    "logical_turn_id": "turn",
    "max_tokens": 128,
    "trace_context": None,
}


def _openai_response(finish_reason: str) -> ChatCompletion:
    return ChatCompletion(
        id="response",
        created=1,
        model="test",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {"name": "reply", "arguments": '{"text":"partial"}'},
                        }
                    ],
                },
            }
        ],
    )


@pytest.mark.parametrize(
    "reason,expected",
    [("length", "length"), ("tool_calls", "toolUse"), ("stop", "toolUse"), ("content_filter", "error")],
)
def test_openai_nonstream_preserves_terminal_reason_with_valid_arguments(reason, expected):
    result, _ = openai_client._default_normal_response_parser(_openai_response(reason), **OPENAI_OPTIONS)
    assert result.tool_calls[0].args == {"text": "partial"}
    assert result.stop_reason == expected
    assert LLMOrchestrator._build_generation_result(result, "test").stop_reason == expected


async def _stream(chunks):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_openai_stream_salvaged_json_stays_length_after_usage_chunk():
    chunks = [
        ChatCompletionChunk(
            id="response",
            created=1,
            model="test",
            object="chat.completion.chunk",
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call",
                                "type": "function",
                                "function": {"name": "reply", "arguments": '{"text":"partial'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        ),
        ChatCompletionChunk(
            id="response",
            created=1,
            model="test",
            object="chat.completion.chunk",
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "length",
                }
            ],
        ),
        ChatCompletionChunk(id="response", created=1, model="test", object="chat.completion.chunk", choices=[]),
    ]
    result, _ = await openai_client._default_stream_response_handler(_stream(chunks), None, **OPENAI_OPTIONS)
    assert result.tool_calls[0].args == {"text": "partial"}
    assert result.stop_reason == "length"


def _gemini_response(reason):
    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    role="model", parts=[Part(function_call=FunctionCall(name="reply", args={"text": "partial"}))]
                ),
                finish_reason=reason,
            )
        ]
    )


@pytest.mark.parametrize(
    "reason,expected",
    [("MAX_TOKENS", "length"), ("CONTINUATION", "length"), ("STOP", "toolUse"), ("SAFETY", "error"), (None, "unknown")],
)
def test_gemini_nonstream_preserves_terminal_reason(reason, expected):
    result, _ = gemini_client._default_normal_response_parser(_gemini_response(reason), logical_turn_id="turn")
    assert result.tool_calls[0].args == {"text": "partial"}
    assert result.stop_reason == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,expected", [("MAX_TOKENS", "length"), ("STOP", "toolUse"), (None, "unknown")])
async def test_gemini_stream_keeps_terminal_signal_after_usage_only_chunk(reason, expected):
    result, _ = await gemini_client._default_stream_response_handler(
        _stream([_gemini_response(reason), GenerateContentResponse(candidates=[])]),
        None,
        logical_turn_id="turn",
    )
    assert result.stop_reason == expected


@pytest.mark.asyncio
async def test_gemini_other_candidate_completion_does_not_authorize_unfinished_tool_call():
    unfinished = _gemini_response(None)
    unfinished.candidates[0].index = 1
    completed_other = GenerateContentResponse(
        candidates=[
            Candidate(
                index=0,
                content=Content(role="model", parts=[Part(text="done")]),
                finish_reason="STOP",
            )
        ]
    )
    result, _ = await gemini_client._default_stream_response_handler(
        _stream([unfinished, completed_other]),
        None,
        logical_turn_id="turn",
    )
    assert result.tool_calls
    assert result.stop_reason == "unknown"


@pytest.mark.parametrize("status,expected", [("completed", "toolUse"), ("", "unknown"), ("in_progress", "unknown")])
def test_responses_requires_explicit_completion(status, expected):
    request = SimpleNamespace(model_info=SimpleNamespace(model_identifier="test"), logical_turn_id="turn")
    response = {
        "id": "response",
        "status": status,
        "output": [
            {
                "type": "function_call",
                "id": "fc",
                "call_id": "call",
                "name": "reply",
                "arguments": '{"text":"partial"}',
            }
        ],
    }
    result, _ = openai_responses_client._parse_completed_response(
        response,
        request,
        "provider",
        "https://example.test/v1",
        ToolArgumentParseMode.STRICT,
    )
    assert result.stop_reason == expected


def test_responses_incomplete_does_not_return_executable_tools():
    request = SimpleNamespace(model_info=SimpleNamespace(model_identifier="test"), logical_turn_id="turn")
    with pytest.raises(RespParseException):
        openai_responses_client._parse_completed_response(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            request,
            "provider",
            "https://example.test/v1",
            ToolArgumentParseMode.STRICT,
        )


def test_unreported_provider_completion_remains_unknown():
    assert APIResponse().stop_reason == "unknown"
    assert LLMResponseResult.from_portable_output(response="text").stop_reason == "unknown"
    assert openai_client._normalize_stop_reason(None, has_tool_calls=True) == "unknown"
