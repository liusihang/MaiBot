from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import asyncio
import json

import pytest

from src.common.data_models.message_component_data_model import TextComponent
from src.common.prompt_i18n import list_prompt_templates, load_prompt
from src.services import memory_flow_service as flow
from src.services import person_fact_writeback as batch


def message(mid="1", user="a", text="我喜欢喝茶", session="group-one", command=False):
    return NS(
        message_id=mid,
        session_id=session,
        platform="qq",
        is_command=command,
        message_info=NS(user_info=NS(user_id=user), group_info=NS(group_id=session)),
        raw_message=NS(components=[TextComponent(text=text)]),
        processed_plain_text=text,
    )


def fact(source, **overrides):
    return {
        "person_id": batch.get_person_id(source.platform, source.message_info.user_info.user_id),
        "fact": "同名用户喜欢喝茶",
        "evidence_message_id": source.message_id,
        "evidence_quote": "我喜欢喝茶",
        "profile_section": "stable_facts",
        **overrides,
    }


@pytest.fixture
def environment(tmp_path, monkeypatch):
    clock = [1000.0]
    sources = {}
    monkeypatch.setattr(batch.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        batch,
        "global_config",
        NS(
            a_memorix=NS(
                integration=NS(person_fact_writeback_enabled=True), filter=NS(enabled=False, mode="whitelist", chats=[])
            )
        ),
    )
    monkeypatch.setattr(batch, "is_bot_self", lambda platform, user: user == "bot")
    monkeypatch.setattr(
        batch,
        "Person",
        lambda person_id: NS(person_id=person_id, is_known=True, person_name="同名用户", nickname="昵称"),
    )
    monkeypatch.setattr(
        batch.PersonFactWritebackService,
        "_load_messages",
        staticmethod(
            lambda session, ids: {
                mid: sources[mid] for mid in ids if mid in sources and sources[mid].session_id == session
            }
        ),
    )
    reader = AsyncMock(return_value={"success": True, "items": []})
    writer = AsyncMock(return_value=NS(success=True, detail="", stored_ids=["stored"], skipped_ids=[]))
    monkeypatch.setattr(batch.memory_service, "fact_admin", reader)
    monkeypatch.setattr(batch.memory_service, "ingest_text", writer)

    def service():
        svc = batch.PersonFactWritebackService(tmp_path / "batches.sqlite3")
        svc._journal.initialize()
        svc._extractor = NS(generate_response=AsyncMock(return_value=NS(response="[]")))
        return svc

    return NS(clock=clock, sources=sources, reader=reader, writer=writer, service=service)


async def enqueue(svc, env, *messages):
    for item in messages:
        env.sources[item.message_id] = item
        await svc.enqueue(item)


@pytest.mark.asyncio
async def test_waits_ten_minutes_and_coalesces_multiple_people(environment):
    e = environment
    svc = e.service()
    await enqueue(svc, e, message("1", "a"), message("2", "b"))
    e.clock[0] += 599
    await svc.run_due()
    svc._extractor.generate_response.assert_not_awaited()
    e.clock[0] += 1
    await svc.run_due()
    svc._extractor.generate_response.assert_awaited_once()
    prompt = svc._extractor.generate_response.call_args.args[0]
    assert batch.get_person_id("qq", "a") in prompt
    assert batch.get_person_id("qq", "b") in prompt
    assert svc._extractor.generate_response.call_args.kwargs["session_id"] == "group-one"


@pytest.mark.asyncio
async def test_duplicate_messages_and_empty_extraction_are_not_examined_again(environment):
    e = environment
    svc = e.service()
    source = message()
    await enqueue(svc, e, source, source, source)
    e.clock[0] += 600
    await svc.run_due()
    await enqueue(svc, e, source)
    e.clock[0] += 600
    await svc.run_due()
    svc._extractor.generate_response.assert_awaited_once()
    assert svc._journal.due_sessions(e.clock[0]) == []


@pytest.mark.asyncio
async def test_no_new_evidence_never_calls_model(environment):
    svc = environment.service()
    environment.clock[0] += 3600
    await svc.run_due()
    svc._extractor.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_messages_respect_previous_attempt_interval(environment):
    e = environment
    svc = e.service()
    await enqueue(svc, e, message())
    e.clock[0] += 600
    await svc.run_due()
    await enqueue(svc, e, message("2"))
    e.clock[0] += 599
    await svc.run_due()
    assert svc._extractor.generate_response.await_count == 1
    e.clock[0] += 1
    await svc.run_due()
    assert svc._extractor.generate_response.await_count == 2


@pytest.mark.asyncio
async def test_restart_recovers_pending_messages_and_persisted_cooldown(environment):
    e = environment
    first = e.service()
    await enqueue(first, e, message())
    e.clock[0] += 300
    restarted = e.service()
    await restarted.run_due()
    restarted._extractor.generate_response.assert_not_awaited()
    e.clock[0] += 300
    await restarted.run_due()
    await enqueue(restarted, e, message("2"))
    again = e.service()
    await again.run_due()
    again._extractor.generate_response.assert_not_awaited()
    e.clock[0] += 600
    await again.run_due()
    again._extractor.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_two_workers_cannot_claim_the_same_chat(environment):
    e = environment
    first, second = e.service(), e.service()
    await enqueue(first, e, message())
    e.clock[0] += 600
    await asyncio.gather(first.run_due(), second.run_due())
    assert first._extractor.generate_response.await_count + second._extractor.generate_response.await_count == 1


@pytest.mark.asyncio
async def test_different_chats_have_separate_batches(environment):
    e = environment
    svc = e.service()
    await enqueue(svc, e, message("1"), message("2", session="group-two"))
    e.clock[0] += 600
    await svc.run_due()
    assert svc._extractor.generate_response.await_count == 2
    assert {c.kwargs["session_id"] for c in svc._extractor.generate_response.call_args_list} == {
        "group-one",
        "group-two",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [message(user="bot"), message(command=True), message(text="")])
async def test_bot_commands_and_nontext_are_not_enqueued(environment, source):
    svc = environment.service()
    await enqueue(svc, environment, source)
    environment.clock[0] += 600
    await svc.run_due()
    svc._extractor.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_chat_filter_is_applied_before_model_call(environment):
    e = environment
    batch.global_config.a_memorix.filter = NS(enabled=True, mode="whitelist", chats=["group:other"])
    svc = e.service()
    await enqueue(svc, e, message())
    e.clock[0] += 600
    await svc.run_due()
    svc._extractor.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_message_overflow_is_left_for_next_window(environment):
    e = environment
    svc = e.service()
    await enqueue(svc, e, *[message(str(i)) for i in range(105)])
    e.clock[0] += 600
    await svc.run_due()
    assert len(svc._journal.jobs("group-one", "pending")) == 5
    await svc.run_due()
    assert svc._extractor.generate_response.await_count == 1
    e.clock[0] += 600
    await svc.run_due()
    assert svc._journal.jobs("group-one", "pending") == []
    assert svc._extractor.generate_response.await_count == 2


@pytest.mark.asyncio
async def test_character_overflow_does_not_launch_extra_calls(environment, monkeypatch):
    e = environment
    monkeypatch.setattr(batch, "BATCH_MAX_CHARS", 12)
    svc = e.service()
    await enqueue(
        svc, e, message("1", text="我喜欢喝茶。"), message("2", text="我喜欢喝茶。"), message("3", text="我喜欢喝茶。")
    )
    e.clock[0] += 600
    await svc.run_due()
    assert len(svc._journal.jobs("group-one", "pending")) == 1
    svc._extractor.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_failure_retries_only_next_ten_minute_window(environment):
    e = environment
    svc = e.service()
    svc._extractor.generate_response.side_effect = RuntimeError("provider unavailable")
    await enqueue(svc, e, message())
    e.clock[0] += 600
    await svc.run_due()
    await svc.run_due()
    assert len(svc._journal.jobs("group-one", "pending")) == 1
    svc._extractor.generate_response.assert_awaited_once()
    e.clock[0] += 600
    await svc.run_due()
    assert svc._extractor.generate_response.await_count == 2


@pytest.mark.asyncio
async def test_malformed_result_does_not_mark_messages_examined(environment):
    e = environment
    svc = e.service()
    svc._extractor.generate_response.return_value.response = "{}"
    await enqueue(svc, e, message())
    e.clock[0] += 600
    await svc.run_due()
    assert len(svc._journal.jobs("group-one", "pending")) == 1


@pytest.mark.asyncio
async def test_same_display_names_keep_real_author_ids_and_stable_fact_metadata(environment):
    e = environment
    svc = e.service()
    a, b = message("1", "a"), message("2", "b")
    svc._extractor.generate_response.return_value.response = json.dumps([fact(a), fact(b)])
    await enqueue(svc, e, a, b)
    e.clock[0] += 600
    await svc.run_due()
    assert e.writer.await_count == 2
    assert {c.kwargs["person_ids"][0] for c in e.writer.call_args_list} == {
        batch.get_person_id("qq", "a"),
        batch.get_person_id("qq", "b"),
    }
    for call in e.writer.call_args_list:
        assert call.kwargs["metadata"]["fact_claim"] == {
            "authority": "direct_user",
            "stability": "stable",
            "profile_section": "stable_facts",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"person_id": "someone-else"},
        {"evidence_message_id": "invented"},
        {"evidence_quote": "我喜欢咖啡"},
    ],
)
async def test_cross_person_and_fabricated_evidence_are_rejected(environment, overrides):
    e = environment
    svc = e.service()
    source = message()
    svc._extractor.generate_response.return_value.response = json.dumps([fact(source, **overrides)])
    await enqueue(svc, e, source)
    e.clock[0] += 600
    await svc.run_due()
    e.writer.assert_not_awaited()
    assert svc._journal.jobs("group-one", "pending") == []


@pytest.mark.asyncio
async def test_existing_fact_is_not_written_again(environment):
    e = environment
    svc = e.service()
    source = message()
    svc._extractor.generate_response.return_value.response = json.dumps([fact(source)])
    e.reader.return_value = {"success": True, "items": [{"value_text": "同名用户喜欢喝茶"}]}
    await enqueue(svc, e, source)
    e.clock[0] += 600
    await svc.run_due()
    e.writer.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        NS(success=False, detail="write failed", stored_ids=[], skipped_ids=[]),
        NS(success=True, detail="chat_filtered", stored_ids=[], skipped_ids=["id"]),
        NS(success=True, detail="", stored_ids=[], skipped_ids=[]),
    ],
)
async def test_failed_persistence_reuses_extraction_even_after_restart(environment, result):
    e = environment
    svc = e.service()
    source = message()
    svc._extractor.generate_response.return_value.response = json.dumps([fact(source)])
    e.writer.return_value = result
    await enqueue(svc, e, source)
    e.clock[0] += 600
    await svc.run_due()
    assert len(svc._journal.jobs("group-one", "extracted")) == 1
    e.writer.return_value = NS(success=True, detail="", stored_ids=["id"], skipped_ids=[])
    restarted = e.service()
    e.clock[0] += 600
    await restarted.run_due()
    restarted._extractor.generate_response.assert_not_awaited()
    assert restarted._journal.jobs("group-one", "extracted") == []


@pytest.mark.asyncio
async def test_only_incoming_messages_trigger_person_fact_queue():
    service = flow.MemoryAutomationService.__new__(flow.MemoryAutomationService)
    service._started = True
    service._observe_sender_identity = AsyncMock()
    service.fact_writeback = NS(enqueue=AsyncMock())
    service.chat_summary_writeback = NS(enqueue=AsyncMock())
    service.image_writeback = NS(enqueue=AsyncMock())
    source = message()
    await service.on_incoming_message(source)
    service.fact_writeback.enqueue.assert_awaited_once_with(source)
    await service.on_message_sent(message("reply", user="bot"))
    service.fact_writeback.enqueue.assert_awaited_once()
    service.chat_summary_writeback.enqueue.assert_awaited_once()


def test_batch_prompt_locales_have_matching_placeholders():
    for locale in ["zh-CN", "en-US", "ja-JP"]:
        prompt = load_prompt("person_fact_batch", locale=locale, people="people-data", messages="message-data")
        assert "people-data" in prompt and "message-data" in prompt
        metadata = list_prompt_templates(locale=locale)["person_fact_batch"].metadata
        assert metadata.display_name and metadata.description


@pytest.mark.asyncio
async def test_numeric_qq_message_id_is_normalized_before_persisting_result(environment):
    e = environment
    svc = e.service()
    source = message("123456789")
    svc._extractor.generate_response.return_value.response = json.dumps([fact(source, evidence_message_id=123456789)])
    await enqueue(svc, e, source)
    e.clock[0] += 600
    await svc.run_due()
    e.writer.assert_awaited_once()
    assert e.writer.call_args.kwargs["metadata"]["evidence_message_ids"] == ["123456789"]
