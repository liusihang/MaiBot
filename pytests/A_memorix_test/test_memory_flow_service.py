import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.A_memorix.core.utils.summary_importer import SUMMARY_PROMPT_TEMPLATE, SummaryImporter
from src.services import memory_flow_service as memory_flow_module


def _fake_global_config(**integration_values):
    return SimpleNamespace(
        a_memorix=SimpleNamespace(
            integration=SimpleNamespace(**integration_values),
        )
    )


def _summary_source_messages(count: int):
    return [
        SimpleNamespace(
            message_id=f"source-{index}", session_id="session-1", timestamp=float(index + 1),
            message_info=SimpleNamespace(
                user_info=SimpleNamespace(user_id="user-1"),
                group_info=SimpleNamespace(group_id="group-1"),
            ),
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_chat_summary_writeback_service_triggers_when_threshold_reached(monkeypatch):
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        memory_flow_module,
        "global_config",
        _fake_global_config(
            chat_summary_writeback_enabled=True,
            chat_summary_writeback_message_threshold=3,
            chat_summary_writeback_context_length=7,
        ),
    )
    monkeypatch.setattr(memory_flow_module, "count_messages", lambda **kwargs: 0 if "before_time" in kwargs else 5)
    monkeypatch.setattr(memory_flow_module, "find_messages", lambda **kwargs: _summary_source_messages(5))

    async def fake_ingest_summary(**kwargs):
        events.append(("ingest_summary", kwargs))
        return SimpleNamespace(success=True, detail="ok", stored_ids=["summary-1"], skipped_ids=[])

    async def fake_load_last_trigger_message_count(self, *, session_id: str, total_message_count: int) -> int:
        del self, session_id, total_message_count
        return 0

    monkeypatch.setattr(memory_flow_module.memory_service, "ingest_summary", fake_ingest_summary)
    monkeypatch.setattr(
        memory_flow_module.ChatSummaryWritebackService,
        "_load_last_trigger_message_count",
        fake_load_last_trigger_message_count,
    )

    service = memory_flow_module.ChatSummaryWritebackService()
    message = SimpleNamespace(session_id="session-1", session=SimpleNamespace(user_id="user-1", group_id="group-1"))

    await service._handle_message(message)

    assert len(events) == 1
    _, payload = events[0]
    assert payload["external_id"] == "chat_auto_summary:v2:session-1:5"
    assert payload["chat_id"] == "session-1"
    assert payload["text"] == ""
    assert payload["metadata"]["generate_from_chat"] is True
    assert payload["metadata"]["context_length"] == 5
    assert payload["metadata"]["writeback_version"] == 2
    assert payload["metadata"]["previous_trigger_message_count"] == 0
    assert payload["metadata"]["pending_message_count"] == 5
    assert payload["metadata"]["trigger"] == "message_threshold"
    assert payload["user_id"] == "user-1"
    assert payload["group_id"] == "group-1"


@pytest.mark.asyncio
async def test_chat_summary_writeback_service_skips_when_threshold_not_reached(monkeypatch):
    called = False

    monkeypatch.setattr(
        memory_flow_module,
        "global_config",
        _fake_global_config(
            chat_summary_writeback_enabled=True,
            chat_summary_writeback_message_threshold=6,
            chat_summary_writeback_context_length=9,
        ),
    )
    monkeypatch.setattr(memory_flow_module, "count_messages", lambda **kwargs: 5)

    async def fake_ingest_summary(**kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(success=True, detail="ok")

    async def fake_load_last_trigger_message_count(self, *, session_id: str, total_message_count: int) -> int:
        del self, session_id, total_message_count
        return 0

    monkeypatch.setattr(memory_flow_module.memory_service, "ingest_summary", fake_ingest_summary)
    monkeypatch.setattr(
        memory_flow_module.ChatSummaryWritebackService,
        "_load_last_trigger_message_count",
        fake_load_last_trigger_message_count,
    )

    service = memory_flow_module.ChatSummaryWritebackService()
    message = SimpleNamespace(session_id="session-1", session=SimpleNamespace(user_id="user-1", group_id="group-1"))

    await service._handle_message(message)

    assert called is False


@pytest.mark.asyncio
async def test_chat_summary_writeback_service_restores_previous_trigger_count(monkeypatch):
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        memory_flow_module,
        "global_config",
        _fake_global_config(
            chat_summary_writeback_enabled=True,
            chat_summary_writeback_message_threshold=3,
            chat_summary_writeback_context_length=7,
        ),
    )
    monkeypatch.setattr(memory_flow_module, "count_messages", lambda **kwargs: 8)
    monkeypatch.setattr(memory_flow_module, "find_messages", lambda **kwargs: _summary_source_messages(8))

    async def fake_ingest_summary(**kwargs):
        events.append(("ingest_summary", kwargs))
        return SimpleNamespace(success=True, detail="ok", stored_ids=["summary-1"], skipped_ids=[])

    async def fake_load_last_trigger_message_count(self, *, session_id: str, total_message_count: int) -> int:
        del self, session_id, total_message_count
        return 5

    monkeypatch.setattr(memory_flow_module.memory_service, "ingest_summary", fake_ingest_summary)
    monkeypatch.setattr(
        memory_flow_module.ChatSummaryWritebackService,
        "_load_last_trigger_message_count",
        fake_load_last_trigger_message_count,
    )

    service = memory_flow_module.ChatSummaryWritebackService()
    message = SimpleNamespace(session_id="session-1", session=SimpleNamespace(user_id="user-1", group_id="group-1"))

    await service._handle_message(message)

    assert len(events) == 1
    _, payload = events[0]
    assert payload["external_id"] == "chat_auto_summary:v2:session-1:8"
    assert service._states["session-1"].last_trigger_message_count == 8


def test_chat_summary_writeback_effective_context_length_uses_pending_window():
    assert (
        memory_flow_module.ChatSummaryWritebackService._effective_context_length(
            configured_context_length=50,
            pending_message_count=12,
        )
        == 12
    )
    assert (
        memory_flow_module.ChatSummaryWritebackService._effective_context_length(
            configured_context_length=8,
            pending_message_count=12,
        )
        == 8
    )


def test_chat_summary_writeback_counts_until_trigger_message(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_count_messages(**kwargs):
        calls.append(kwargs)
        return 12

    monkeypatch.setattr(memory_flow_module, "count_messages", fake_count_messages)

    assert (
        memory_flow_module.ChatSummaryWritebackService._count_messages_until_trigger(
            session_id="session-1",
            message_time=123.45,
        )
        == 12
    )
    assert calls == [{"session_id": "session-1", "end_time": 123.45}]


def test_summary_prompt_keeps_static_rules_before_chat_history():
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        bot_name="测试机器人",
        personality_context="你的性格设定是：稳定。",
        identity_catalog="[{id:1}]",
        previous_summary_context="上轮动态摘要",
        chat_history="用户：第一条动态消息",
        image_evidence_catalog="无",
    )

    rules_index = prompt.index("先筛选")
    chat_history_index = prompt.index("本轮聊天")
    dynamic_history_index = prompt.index("用户：第一条动态消息")

    assert rules_index < chat_history_index < dynamic_history_index
    assert rules_index < prompt.index("上轮动态摘要")
    assert rules_index < prompt.index("你的性格设定是：稳定。")
    assert rules_index < prompt.index("[{id:1}]")


def test_summary_prompt_forbids_repeating_rejected_fact_values():
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        bot_name="测试机器人",
        personality_context="你的性格设定是：稳定。",
        identity_catalog="[]",
        previous_summary_context="",
        chat_history="用户：不是猫毛，是青霉素",
        image_evidence_catalog="无",
    )

    assert "更正只写最终新状态" in prompt
    assert "不复述错误旧值和纠错过程" in prompt
    assert "不复述其具体值" in prompt


def test_summary_review_cleaner_does_not_guess_validity_from_keywords():
    dirty_summary = "此前记录林遥对花生过敏，这是测试示例，后来已纠正。"

    assert SummaryImporter._clean_review_summary(dirty_summary) == dirty_summary


@pytest.mark.asyncio
async def test_chat_summary_writeback_service_falls_back_to_current_count_for_legacy_summary(monkeypatch):
    called = False

    monkeypatch.setattr(
        memory_flow_module,
        "global_config",
        _fake_global_config(
            chat_summary_writeback_enabled=True,
            chat_summary_writeback_message_threshold=3,
            chat_summary_writeback_context_length=7,
        ),
    )
    monkeypatch.setattr(memory_flow_module, "count_messages", lambda **kwargs: 5)

    async def fake_ingest_summary(**kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(success=True, detail="ok")

    async def fake_load_last_trigger_message_count(self, *, session_id: str, total_message_count: int) -> int:
        del self, session_id, total_message_count
        return 5

    monkeypatch.setattr(memory_flow_module.memory_service, "ingest_summary", fake_ingest_summary)
    monkeypatch.setattr(
        memory_flow_module.ChatSummaryWritebackService,
        "_load_last_trigger_message_count",
        fake_load_last_trigger_message_count,
    )

    service = memory_flow_module.ChatSummaryWritebackService()
    message = SimpleNamespace(session_id="session-1", session=SimpleNamespace(user_id="user-1", group_id="group-1"))

    await service._handle_message(message)

    assert called is False
    assert service._states["session-1"].last_trigger_message_count == 5


@pytest.mark.asyncio
async def test_chat_summary_writeback_service_loads_trigger_count_from_summary_metadata(monkeypatch):
    class FakeMetadataStore:
        @staticmethod
        def query(sql: str, params: tuple):
            assert "chat_summary_checkpoints" in sql
            assert params == ("session-1", "chat_auto_summary:v2:session-1:%")
            return [{"count": 0}]

        @staticmethod
        def get_paragraphs_by_source(source: str):
            assert source == "chat_summary:session-1"
            return [
                {"created_at": 1.0, "metadata": {"trigger_message_count": 3, "writeback_version": 2}},
                {"created_at": 2.0, "metadata": {"trigger_message_count": 6, "writeback_version": 2}},
                {"created_at": 3.0, "metadata": {"trigger_message_count": 8}},
            ]

    class FakeRuntimeManager:
        @staticmethod
        async def _ensure_kernel():
            return SimpleNamespace(metadata_store=FakeMetadataStore())

    monkeypatch.setattr(memory_flow_module.memory_service_module, "a_memorix_host_service", FakeRuntimeManager())

    service = memory_flow_module.ChatSummaryWritebackService()

    restored = await service._load_last_trigger_message_count(session_id="session-1", total_message_count=8)

    assert restored == 6


@pytest.mark.asyncio
async def test_memory_automation_service_auto_starts_and_delegates(monkeypatch):
    monkeypatch.setattr(memory_flow_module.ImageMemoryWritebackService, 'start', AsyncMock())
    monkeypatch.setattr(memory_flow_module.ImageMemoryWritebackService, 'shutdown', AsyncMock())
    observe_identity = AsyncMock()
    monkeypatch.setattr(memory_flow_module.MemoryAutomationService, "_observe_sender_identity", observe_identity)
    events: list[tuple[str, str]] = []

    class FakeFactWriteback:
        async def start(self):
            events.append(("start", "fact"))

        async def enqueue(self, message):
            events.append(("sent", message.session_id))

        async def shutdown(self):
            events.append(("shutdown", "fact"))

    class FakeChatSummaryWriteback:
        async def start(self):
            events.append(("start", "summary"))

        async def enqueue(self, message):
            events.append(("summary", message.session_id))

        async def shutdown(self):
            events.append(("shutdown", "summary"))

    service = memory_flow_module.MemoryAutomationService()
    service.fact_writeback = FakeFactWriteback()
    service.chat_summary_writeback = FakeChatSummaryWriteback()

    await service.on_message_sent(SimpleNamespace(session_id="session-1"))
    await service.shutdown()

    observe_identity.assert_awaited_once()
    assert events == [
        ("start", "fact"),
        ("start", "summary"),
        ("summary", "session-1"),
        ("shutdown", "summary"),
        ("shutdown", "fact"),
    ]


@pytest.mark.asyncio
async def test_memory_automation_service_incoming_message_enqueues_person_facts(monkeypatch):
    monkeypatch.setattr(memory_flow_module.ImageMemoryWritebackService, 'start', AsyncMock())
    monkeypatch.setattr(memory_flow_module.ImageMemoryWritebackService, 'enqueue', AsyncMock())
    monkeypatch.setattr(memory_flow_module.ImageMemoryWritebackService, 'shutdown', AsyncMock())
    observe_identity = AsyncMock()
    monkeypatch.setattr(memory_flow_module.MemoryAutomationService, "_observe_sender_identity", observe_identity)
    events: list[tuple[str, str]] = []

    class FakeFactWriteback:
        async def start(self):
            events.append(("start", "fact"))

        async def enqueue(self, message):
            events.append(("sent", message.session_id))

        async def shutdown(self):
            events.append(("shutdown", "fact"))

    class FakeChatSummaryWriteback:
        async def start(self):
            events.append(("start", "summary"))

        async def enqueue(self, message):
            events.append(("summary", message.session_id))

        async def shutdown(self):
            events.append(("shutdown", "summary"))

    service = memory_flow_module.MemoryAutomationService()
    service.fact_writeback = FakeFactWriteback()
    service.chat_summary_writeback = FakeChatSummaryWriteback()

    await service.on_incoming_message(SimpleNamespace(session_id="session-1"))
    await service.shutdown()

    observe_identity.assert_awaited_once()
    assert events == [
        ("start", "fact"),
        ("start", "summary"),
        ("sent", "session-1"),
        ("shutdown", "summary"),
        ("shutdown", "fact"),
    ]


def test_chat_summary_writeback_metadata_decodes_json_bytes_only():
    metadata = memory_flow_module.ChatSummaryWritebackService._paragraph_metadata(
        {"metadata": json.dumps({"trigger_message_count": 7}).encode("utf-8")}
    )
    invalid = memory_flow_module.ChatSummaryWritebackService._paragraph_metadata({"metadata": b"not-json-or-utf8"})

    assert metadata == {"trigger_message_count": 7}
    assert invalid == {}
