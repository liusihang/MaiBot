"""Exercise the real search adapter with local, deterministic transport doubles only."""

from types import SimpleNamespace
import json

import pytest

from deploy.qqbot.plugins.gpt_web_search import openwebui_search as search_module
from deploy.qqbot.plugins.gpt_web_search.plugin import Config, GPTWebSearchPlugin, ReadSearchArguments, SearchArguments
from src.plugin_runtime.component_query import ComponentQueryService
from src.plugin_runtime.host.component_registry import ComponentTypes, ToolEntry


class FakeSearchTransport:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def __call__(self, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, endpoint, **kwargs):
        self.requests.append(kwargs["json"])
        return self

    status = 200

    async def json(self, **kwargs):
        return self.payload


def make_plugin(monkeypatch, payload, limit=4000):
    transport = FakeSearchTransport(payload)
    monkeypatch.setattr(search_module.aiohttp, "ClientSession", transport)
    plugin = GPTWebSearchPlugin()
    plugin._plugin_config_instance = Config.model_validate(
        {"search": {"api_key": "secret-test-key", "max_output_chars": limit}}
    )
    plugin._configure()
    return plugin, transport


def history_result(result):
    entry = ToolEntry(name="gpt_web_search", component_type=ComponentTypes.TOOL.value, plugin_id="test", metadata={})
    return ComponentQueryService._parse_tool_invoke_result(entry, result)


@pytest.mark.asyncio
async def test_full_tail_and_structured_evidence_stored_once_history_is_bounded(monkeypatch):
    payload = {
        "output": "opening evidence\n" + "前言😀" * 9000 + "\nTAIL-FACT: deadline 2040-01-01 https://example.org/tail",
        "results": [
            {
                "url": "https://example.org/first",
                "ref_id": "turn0search0",
                "title": "First",
                "content": "structured-only " * 9000,
            },
            {
                "url": "https://example.org/last",
                "ref_id": "turn0search1",
                "title": "Last",
                "description": "FINAL-STRUCTURED-FACT",
            },
        ],
        "sources": [{"url": "https://example.org/raw", "other": "unique-raw-evidence" * 800}],
        "encrypted_output": "not-model-evidence",
    }
    plugin, transport = make_plugin(monkeypatch, payload)
    result = await plugin.search_web(
        query="deadline", count=1, stream_id="actual-session-a", group_id="a", platform="qq"
    )
    parsed = json.loads(result["content"])
    assert result["success"] and parsed["truncated"]
    assert result["tool_result_preview"] and result["reader_tool"] == "gpt_read_search_result"
    assert len(history_result(result).get_history_content()) <= 4000
    assert "TAIL-FACT" not in result["content"]
    assert len(plugin._search._stored_results) == 1
    stored = next(iter(plugin._search._stored_results.values()))
    original = json.loads(stored)
    assert original["output"] == payload["output"]
    assert original["results"] == payload["results"]
    assert original["sources"] == payload["sources"]
    assert "encrypted_output" not in original

    for keyword in ("TAIL-FACT", "FINAL-STRUCTURED-FACT", "unique-raw-evidence"):
        read = await plugin.read_search_result(
            result_id=parsed["result_id"], keyword=keyword, stream_id="actual-session-a", group_id="a", platform="qq"
        )
        assert read["success"] and keyword in json.loads(read["content"])["content"]
        assert len(history_result(read).get_history_content()) <= 4000
    assert len(transport.requests) == 1, "Reading cached evidence must not call a network/model API"
    assert len(plugin._search._stored_results) == 1, "A read must not store another copy"


@pytest.mark.asyncio
async def test_read_pages_reconstruct_raw_and_preserve_unicode_offsets(monkeypatch):
    payload = {"output": "İ😀甲乙\n" * 1700 + "TAIL", "results": []}
    plugin, transport = make_plugin(monkeypatch, payload, limit=1000)
    result = await plugin.search_web(query="page", stream_id="s")
    receipt = json.loads(result["content"])
    assert len(result["content"]) <= 1000
    offset, chunks = 0, []
    while offset is not None:
        response = await plugin.read_search_result(
            result_id=receipt["result_id"], offset=offset, max_chars=2800, stream_id="s"
        )
        page = json.loads(response["content"])
        assert response["success"] and len(response["content"]) <= 1000
        assert page["offset"] == offset
        chunks.append(page["content"])
        offset = page["next_offset"]
    assert "".join(chunks) == next(iter(plugin._search._stored_results.values()))
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_scopes_and_request_scope_cannot_read_another_conversation(monkeypatch):
    plugin, transport = make_plugin(monkeypatch, {"output": "PRIVATE-FACT"})
    first = await plugin.search_web(query="fact", stream_id="a", platform="qq", group_id="1")
    result_id = json.loads(first["content"])["result_id"]
    for scope in (
        {"stream_id": "b", "platform": "qq", "group_id": "1"},
        {"stream_id": "a", "platform": "qq", "group_id": "2"},
    ):
        read = await plugin.read_search_result(result_id=result_id, **scope)
        assert not read["success"] and "PRIVATE-FACT" not in read["content"]
        assert json.loads(read["content"])["error"]["code"] == "result_unavailable"
    missing = await plugin.search_web(query="no session")
    assert not missing["success"] and len(transport.requests) == 1
    tools = plugin._search
    request = SimpleNamespace(state=SimpleNamespace())
    other_request = SimpleNamespace(state=SimpleNamespace())
    first = await tools.search_web(query="fact", __user__={"id": "a"}, __request__=request)
    same = await tools.read_search_result(first["result_id"], __user__={"id": "a"}, __request__=request)
    other = await tools.read_search_result(first["result_id"], __user__={"id": "a"}, __request__=other_request)
    other_user = await tools.read_search_result(first["result_id"], __user__={"id": "b"}, __request__=request)
    assert same["status"] == "ok"
    assert other["error"]["code"] == other_user["error"]["code"] == "result_unavailable"


@pytest.mark.asyncio
async def test_open_click_find_keep_upstream_scope_and_refs(monkeypatch):
    plugin, transport = make_plugin(
        monkeypatch, {"output": "Source (https://example.org/page)\n【turn0search0】\nL1: evidence"}
    )
    initial = await plugin.search_web(query="source", stream_id="s", platform="qq", group_id="g")
    assert initial["success"] and "turn0search0" in initial["content"]
    for command in (
        {"open": [{"ref_id": "turn0search0", "lineno": 1}]},
        {"click": [{"ref_id": "turn0search0", "id": 7}]},
        {"find": [{"ref_id": "turn0search0", "pattern": "evidence"}]},
    ):
        followup = await plugin.search_web(commands=command, stream_id="s", platform="qq", group_id="g")
        assert followup["success"]
        assert transport.requests[-1]["commands"] == command
    assert len({request["id"] for request in transport.requests}) == 1


@pytest.mark.asyncio
async def test_eviction_restart_and_config_reload_are_explicit(monkeypatch):
    plugin, _ = make_plugin(monkeypatch, {"output": "original"})
    result = await plugin.search_web(query="fact", stream_id="s")
    result_id = json.loads(result["content"])["result_id"]
    plugin.config.search.api_key = "rotated-key"
    plugin._configure()
    assert (await plugin.read_search_result(result_id=result_id, stream_id="s"))["success"]
    restarted = GPTWebSearchPlugin()
    assert not (await restarted.read_search_result(result_id=result_id, stream_id="s"))["success"]
    for _ in range(search_module.MAX_STORED_RESULTS_PER_SCOPE):
        await plugin.search_web(query="fact", stream_id="s")
    expired = await plugin.read_search_result(result_id=result_id, stream_id="s")
    assert not expired["success"] and "expired" in expired["content"]


@pytest.mark.asyncio
async def test_oversized_raw_is_not_silently_truncated_and_secrets_not_stored(monkeypatch):
    plugin, transport = make_plugin(monkeypatch, {"output": "x" * search_module.MAX_STORED_RESULT_CHARS})
    response = await plugin.search_web(query="big", stream_id="s")
    assert not response["success"] and "result_too_large" in response["content"]
    assert not plugin._search._stored_results
    transport.payload = {"output": "not a credential: secret-test-key", "encrypted_content": "opaque"}
    response = await plugin.search_web(query="small", stream_id="s")
    assert response["success"]
    stored = next(iter(plugin._search._stored_results.values()))
    assert "secret-test-key" not in stored and "opaque" not in stored
    assert "[REDACTED]" in stored


@pytest.mark.asyncio
async def test_reader_rejects_unbounded_keyword_and_context_is_not_in_model_schema(monkeypatch):
    plugin, _ = make_plugin(monkeypatch, {"output": "evidence"})
    result = await plugin.search_web(query="fact", stream_id="s")
    result_id = json.loads(result["content"])["result_id"]
    for arguments in ({"keyword": "x" * 100_000}, {"max_chars": 99999}, {"offset": True}):
        read = await plugin.read_search_result(result_id=result_id, stream_id="s", **arguments)
        assert not read["success"] and len(read["content"]) < 500
    for schema in (SearchArguments.model_json_schema(), ReadSearchArguments.model_json_schema()):
        assert schema["additionalProperties"] is False
        assert "stream_id" not in schema["properties"]
