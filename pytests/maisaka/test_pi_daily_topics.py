"""Real official Pi SDK, local mock wire only; no QQ or paid model calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import json

import pytest

from pytests.maisaka.test_pi_harness_host import engine, history, wire_server
from pytests.services.test_daily_topic_service import record, stamp, topic
from src.config.config import global_config
from src.config.model_configs import APIProvider, ModelInfo, ModelTaskConfig, TaskConfig
from src.maisaka.pi_runtime.runner import PiGroupRunner
from src.services.daily_topics.config import DailyTopicsConfig
from src.services.daily_topics.core import DailyTopicProcessor
from src.services.daily_topics.service import DailyTopicService, PiTopicSummarizer
from src.services.pi_harness import config as pi_config, inference
from src.services.pi_harness.client import PiHarnessClient


@pytest.mark.asyncio
async def test_daily_board_keeps_system_and_prefix_dedups_restarts_and_reinjects_after_compaction(
    monkeypatch, tmp_path
):
    service = DailyTopicService(tmp_path / "topics.sqlite3")
    store = await service.store()
    store.ingest_page([record(session="real-chat")], 1)
    await DailyTopicProcessor(store, AsyncMock(return_value=topic())).process(
        "real-chat", "2026-10-29", stamp("2026-10-29T00:00:00"), stamp("2026-10-29T03:00:00")
    )
    monkeypatch.setattr(global_config.daily_topics, "enabled", True)
    monkeypatch.setattr("src.services.daily_topics.service.daily_topic_service", service)
    monkeypatch.setattr("src.maisaka.pi_runtime.runner.build_references", AsyncMock(return_value=[]))
    with wire_server() as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        host._runtime._chat_history.append(history("unrelated live message, preserve verbatim", "m1"))
        runner = PiGroupRunner(host, data_dir=tmp_path / "pi")
        try:
            await runner.run([])
            before = requests[-1]["messages"]
            system = before[0]
            assert system["role"] == "system" and "daily_topic_board" not in json.dumps(system)
            assert "daily_topic_board" in json.dumps(before)
            host._runtime._chat_history.append(history("a new daytime input", "m2"))
            await runner.run([])
            assert requests[-1]["messages"][: len(before)] == before
            assert (await runner.bridge.request("state", sessionId=runner.session_id))["visibleReferenceCount"] == 1
            # A clearing publication is an explicit new board, not absence.
            await DailyTopicProcessor(store, AsyncMock(return_value=[])).process(
                "real-chat", "2026-10-30", stamp("2026-10-30T00:00:00"), stamp("2026-10-30T03:00:00")
            )
            before = requests[-1]["messages"]
            await runner.run([])
            assert requests[-1]["messages"][: len(before)] == before
            assert requests[-1]["messages"][0] == system
            assert "cleared_no_active_topics" in json.dumps(requests[-1]["messages"])
            assert (await runner.bridge.request("state", sessionId=runner.session_id))["visibleReferenceCount"] == 2
            await runner.close()
            runner = PiGroupRunner(host, data_dir=tmp_path / "pi")
            await runner.run([])
            state = await runner.bridge.request("state", sessionId=runner.session_id)
            assert state["visibleReferenceCount"] == 2  # reopen does not repeat visible version
            compacted = await runner.bridge.request(
                "compact",
                sessionId=runner.session_id,
                expectedRevision=state["revision"],
                summary="Prior live conversation.",
            )
            assert compacted["visibleReferenceCount"] == 0
            await runner.run([])
            assert (await runner.bridge.request("state", sessionId=runner.session_id))["visibleReferenceCount"] == 1
            last = json.dumps(requests[-1]["messages"])
            assert "cleared_no_active_topics" in last and "Alice discussed plans" not in last
            assert requests[-1]["messages"][0] == system
            assert invoked == []
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_daily_source_read_uses_existing_scoped_tool_without_model_or_qq(monkeypatch, tmp_path):
    service = DailyTopicService(tmp_path / "topics.sqlite3")
    store = await service.store()
    store.ingest_page([record(session="real-chat"), record("2", "other", "private-other-group")], 2)
    monkeypatch.setattr("src.services.daily_topics.service.daily_topic_service", service)
    with wire_server() as (config, requests):
        monkeypatch.setattr("src.maisaka.pi_runtime.runner.export_harness_config", lambda: config)
        host, invoked = engine()
        runner = PiGroupRunner(host, data_dir=tmp_path / "pi")
        try:
            await runner.ensure_initialized()
            for number, source in enumerate(("1", "2")):
                value = await runner._callback(
                    "tool",
                    {
                        "sessionId": "real-chat",
                        "toolCallId": "read-" + source,
                        "name": "read_thread_archive",
                        "arguments": {"event_ids": ["daily-topic-source:message:" + source]},
                    },
                )
                text = json.dumps(value)
                assert "private-other-group" not in text
                if number == 0:
                    assert "qq-1" in text
            assert requests == [] and invoked == []
        finally:
            await runner.close()


@pytest.mark.asyncio
async def test_daily_summary_runs_through_real_pi_sdk_and_no_tools(monkeypatch):
    with wire_server([{"text": "[]"}]) as (raw, requests):
        tasks = ModelTaskConfig(utils=TaskConfig(model_list=["mock"], max_tokens=1000))
        configured = SimpleNamespace(
            models=[ModelInfo(**raw["models"][0])],
            api_providers=[APIProvider(**raw["api_providers"][0])],
            model_task_config=tasks,
        )
        manager = SimpleNamespace(get_model_config=lambda: configured)
        monkeypatch.setattr(pi_config, "config_manager", manager)
        monkeypatch.setattr(inference, "config_manager", manager)
        monkeypatch.setattr(inference.PiInferenceClient, "_record_usage", AsyncMock())
        client = PiHarnessClient()
        monkeypatch.setattr(inference, "get_harness_client", lambda: client)
        try:
            result = await PiTopicSummarizer(DailyTopicsConfig())([], [record()], {"session_id": "real-chat"})
            assert result == [] and len(requests) == 1
            assert not requests[0].get("tools")
            assert requests[0]["messages"][0]["role"] == "system"
            assert "source_parts" in json.dumps(requests[0]["messages"][1:])
        finally:
            await client.close()
