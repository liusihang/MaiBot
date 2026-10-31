"""Offline scheduling, shared extraction and source-bound review regressions."""

from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import asyncio
import json

import pytest

from src.learners import joint_learning as joint
from src.learners.joint_learning_journal import JointLearningJournal


def journal(tmp_path):
    result = JointLearningJournal(tmp_path / "learning.sqlite")
    result.initialize()
    return result


def test_time_or_count_and_restart(tmp_path):
    state = journal(tmp_path)
    state.enqueue("chat", ["m1"], 100)
    assert state.next_batch("chat", 1899) is None
    batch = state.next_batch("chat", 1900)
    assert json.loads(batch["source_ids"]) == ["m1"]
    state.save(batch["id"], extraction='{"expressions":[],"jargon":[]}', review="[]", expressions_done=1)
    restored = JointLearningJournal(state.path)
    assert restored.next_batch("chat", 1901)["expressions_done"] == 1
    state.save(batch["id"], done=1)
    state.enqueue("chat", ["m1"], 2000)
    assert state.next_batch("chat", 9000) is None


def test_count_drains_backlog_without_wait_or_drop(tmp_path):
    state = journal(tmp_path)
    state.enqueue("chat", [str(i) for i in range(250)], 100)
    first = state.next_batch("chat", 100)
    assert len(json.loads(first["source_ids"])) == 100
    state.save(first["id"], done=1)
    second = state.next_batch("chat", 100)
    assert len(json.loads(second["source_ids"])) == 100
    state.save(second["id"], done=1)
    assert state.next_batch("chat", 1899) is None
    assert len(json.loads(state.next_batch("chat", 1900)["source_ids"])) == 50


def test_retry_keeps_model_results_and_scope(tmp_path):
    state = journal(tmp_path)
    state.enqueue("a", ["m"], 0)
    state.enqueue("b", ["m"], 100)
    batch = state.next_batch("a", 1800)
    state.save(batch["id"], extraction="saved", retry_at=2100)
    assert state.next_batch("a", 2099) is None
    assert state.next_batch("a", 2100)["extraction"] == "saved"
    assert state.next_batch("b", 1800) is None
    with pytest.raises(ValueError):
        state.save(batch["id"], source_ids="bad")


@pytest.mark.parametrize(
    "obj",
    [
        [],
        {},
        {"expressions": [], "jargon": [], "x": 1},
        {"expressions": [{"situation": "s", "style": "v", "id": "2"}], "jargon": []},
        {"expressions": [], "jargon": [{"content": "v", "id": 1}]},
    ],
)
def test_strict_extraction(obj):
    with pytest.raises(ValueError):
        joint.parse_extraction(json.dumps(obj), 1)


def test_multiple_candidates_and_review_schema():
    values = {"expressions": [{"situation": "s", "style": str(i), "id": "1"} for i in range(12)], "jargon": []}
    assert len(joint.parse_extraction(json.dumps(values), 1)["expressions"]) == 12
    assert joint.parse_review('[{"id":0,"suitable":true,"reason":"fits"}]', 1) == [0]
    for text in ["[]", '[{"id":0,"suitable":"false","reason":"bad"}]', '[{"id":true,"suitable":true,"reason":"bad"}]']:
        with pytest.raises(ValueError):
            joint.parse_review(text, 1)


def coordinator(tmp_path):
    runtime = NS(
        session_id="chat",
        _enable_expression_learning=True,
        _enable_jargon_learning=True,
        _expression_learner=NS(),
        _jargon_learner=NS(_run_learning_batch=AsyncMock()),
        _jargon_miner=NS(),
    )
    result = joint.JointLearningCoordinator(runtime, tmp_path / "state.sqlite")
    result.journal.initialize()
    result.initialized = True
    return result


@pytest.mark.asyncio
async def test_single_model_extraction_reused_after_write_failure(tmp_path, monkeypatch):
    service = coordinator(tmp_path)
    service.journal.enqueue("chat", ["m"], 0)
    batch = service.journal.next_batch("chat", 1800)
    message = NS(message_id="m", raw_message="xyz", message_info=NS(user_info=NS(user_id="u", user_nickname="User")))
    monkeypatch.setattr(service, "load_sources", lambda *args: [message])
    monkeypatch.setattr(joint, "extract_learning_text", lambda raw: raw)
    monkeypatch.setattr(joint.JargonLearner, "_build_source_item_from_session_message", lambda message: message)
    monkeypatch.setattr(joint, "load_prompt", lambda *args, **kwargs: json.dumps(kwargs))
    service.client = NS(
        generate_response=AsyncMock(
            return_value=NS(
                response=json.dumps(
                    {
                        "expressions": [{"situation": "s", "style": "xyz", "id": "1"}],
                        "jargon": [{"content": "xyz", "id": "1"}, {"content": "invented", "id": "1"}],
                    }
                )
            )
        )
    )
    service.process_expressions = AsyncMock(side_effect=RuntimeError("db down"))
    with pytest.raises(RuntimeError):
        await service.process(batch)
    batch = service.journal.next_batch("chat", 1801)
    assert batch["extraction"] is not None
    service.process_expressions = AsyncMock()
    await service.process(batch)
    assert service.client.generate_response.await_count == 1
    args = service.runtime._jargon_learner._run_learning_batch.call_args.kwargs
    assert json.loads(args["extracted_response"]) == [{"content": "xyz", "id": "1"}]
    assert service.journal.next_batch("chat", 1802) is None


@pytest.mark.asyncio
async def test_review_dedup_and_shared_evidence_checkpoint(tmp_path, monkeypatch):
    service = coordinator(tmp_path)
    learner = NS(
        _prepare_learning_messages=lambda m: m,
        _filter_expressions=lambda c, m: c,
        _find_similar_expression=lambda *a, **k: None,
        _run_learning_batch=AsyncMock(),
    )
    service.runtime._expression_learner = learner
    service.review_client = NS(
        generate_response=AsyncMock(
            return_value=NS(
                response='[{"id":0,"suitable":true,"reason":"fits"},{"id":1,"suitable":false,"reason":"unclear"}]'
            )
        )
    )
    captured = {}

    def prompt(name, **kwargs):
        captured.update(json.loads(kwargs["candidates"]))
        return "prompt"

    monkeypatch.setattr(joint, "load_prompt", prompt)
    monkeypatch.setattr(joint, "extract_learning_text", lambda raw: raw)
    monkeypatch.setattr(joint.global_config.expression, "expression_self_reflect", True)
    monkeypatch.setattr(
        service,
        "review_window",
        lambda *a: {
            "before_count": 10,
            "after_count": 0,
            "complete": False,
            "messages": [{"message_id": "m", "text": "x", "is_source": True}],
        },
    )
    service.journal.enqueue("chat", ["m"], 0)
    batch = service.journal.next_batch("chat", 1800)
    candidates = [
        {"situation": "s", "style": "x", "id": "1"},
        {"situation": "s", "style": "x", "id": "1"},
        {"situation": "s2", "style": "y", "id": "1"},
    ]
    await service.process_expressions(batch, [NS(message_id="m", platform="qq", raw_message="x")], candidates)
    assert len(captured["messages"]) == 1
    assert len(captured["candidates"]) == 2
    assert captured["candidates"][0]["context"]["after_count"] == 0
    batch = service.journal.next_batch("chat", 1800)
    assert len(json.loads(batch["review"])) == 1
    await service.process_expressions(batch, [NS(message_id="m", platform="qq", raw_message="x")], candidates)
    assert service.review_client.generate_response.await_count == 1
    assert learner._run_learning_batch.call_args.kwargs["already_reviewed"] is True


@pytest.mark.asyncio
async def test_timer_drains_due_persisted_batch_without_new_message(tmp_path, monkeypatch):
    service = coordinator(tmp_path)
    service.journal.enqueue("chat", ["m"], 0)
    called = asyncio.Event()

    async def process(batch):
        service.journal.save(batch["id"], done=1)
        called.set()

    service.process = process
    monkeypatch.setattr(joint.expression_learning_batch_gate, "acquire", AsyncMock(return_value=NS(acquired=True)))
    monkeypatch.setattr(joint.expression_learning_batch_gate, "release", AsyncMock())
    await service.start()
    # Verify the persisted scheduler wakes without another message, rather than
    # requiring thread-pool dispatch and SQLite I/O to finish in one second.
    await asyncio.wait_for(called.wait(), 10)
    await service.stop()
    assert service.task is None


@pytest.mark.asyncio
async def test_disabled_categories_make_no_call(tmp_path, monkeypatch):
    service = coordinator(tmp_path)
    service.runtime._enable_expression_learning = False
    service.runtime._enable_jargon_learning = False
    monkeypatch.setattr(service, "start", AsyncMock())
    await service.collect([NS(original_message=NS())])
    service.start.assert_not_awaited()


def test_prompts_aligned_and_runtime_uses_joint_timer():
    root = Path(__file__).parents[2]
    for locale in ["zh-CN", "en-US", "ja-JP"]:
        assert "{messages}" in (root / "prompts" / locale / "learn_joint.prompt").read_text()
        assert "{candidates}" in (root / "prompts" / locale / "expression_batch_review.prompt").read_text()
    source = (root / "src/maisaka/runtime.py").read_text()
    assert "await self._joint_learning.start()" in source
    assert "await self._joint_learning.stop()" in source
    assert "await self._joint_learning.collect(sources)" in source


def test_review_window_has_ten_actual_neighbours_and_source(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from datetime import datetime
    from sqlmodel import Session, create_engine

    engine = create_engine(f"sqlite:///{tmp_path / 'messages.sqlite'}")
    joint.Messages.__table__.create(engine)
    with Session(engine) as db:
        for i in range(35):
            db.add(
                joint.Messages(
                    message_id=str(i),
                    session_id="chat",
                    platform="qq",
                    user_id="u",
                    user_nickname="user",
                    timestamp=datetime(2026, 1, 1),
                    raw_content=b"x",
                    processed_plain_text=f"text{i}",
                    reply_to="4" if i == 15 else None,
                )
            )
        db.add(
            joint.Messages(
                message_id="other",
                session_id="other",
                platform="qq",
                user_id="u",
                user_nickname="user",
                timestamp=datetime(2026, 1, 1),
                raw_content=b"x",
            )
        )
        db.commit()

    @contextmanager
    def session(**kwargs):
        with Session(engine) as db:
            yield db

    monkeypatch.setattr(joint, "get_db_session", session)
    monkeypatch.setattr(joint.SessionMessage, "from_db_instance", lambda row: NS(raw_message=row.processed_plain_text))
    monkeypatch.setattr(joint, "extract_learning_text", lambda raw: raw or "")
    window = joint.JointLearningCoordinator.review_window("chat", "15")
    assert window["complete"] is True
    assert [m["message_id"] for m in window["messages"]] == [str(i) for i in range(5, 26)]
    assert [m["message_id"] for m in window["messages"] if m["is_source"]] == ["15"]
    assert window["messages"][10]["reply_to_message_id"] == "4"
    tail = joint.JointLearningCoordinator.review_window("chat", "34")
    assert tail["after_count"] == 0 and tail["complete"] is False
    assert all(m["message_id"] != "other" for m in tail["messages"])


@pytest.mark.asyncio
async def test_reviewed_partial_write_retry_does_not_increment_existing(tmp_path, monkeypatch):
    from src.learners import expression_learner as module

    learner = module.ExpressionLearner("chat")
    monkeypatch.setattr(module.prompt_manager, "get_prompt", lambda name: NS(add_context=lambda *a: None))
    monkeypatch.setattr(module.prompt_manager, "render_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(learner, "_prepare_learning_messages", lambda messages: messages)
    monkeypatch.setattr(
        learner, "_get_runtime_manager", lambda: NS(invoke_hook=AsyncMock(return_value=NS(aborted=False, kwargs={})))
    )
    monkeypatch.setattr(learner, "_filter_expressions", lambda candidates, messages: [(a, b) for a, b, c in candidates])
    monkeypatch.setattr(learner, "_get_session_display_name", lambda sid: sid)
    monkeypatch.setattr(learner, "_sync_expression_vector_index_batch", AsyncMock())
    stored, attempts = {}, []
    monkeypatch.setattr(
        learner, "_find_similar_expression", lambda a, b, **kw: (stored[b], 1.0) if b in stored else None
    )

    async def upsert(situation, style, **kwargs):
        attempts.append(style)
        if style == "second" and attempts.count("second") == 1:
            return None
        stored[style] = NS(item_id=len(stored) + 1)
        return stored[style]

    monkeypatch.setattr(learner, "_upsert_expression_to_db", upsert)
    response = json.dumps([{"situation": "s", "style": style, "id": "1"} for style in ["first", "second"]])
    with pytest.raises(RuntimeError):
        await learner._run_learning_batch(
            [], learning_session_id="chat", learned_contents=[], extracted_response=response, already_reviewed=True
        )
    await learner._run_learning_batch(
        [], learning_session_id="chat", learned_contents=[], extracted_response=response, already_reviewed=True
    )
    assert attempts == ["first", "second", "second"]
    assert len(learner._sync_expression_vector_index_batch.call_args.args[0]) == 2


def test_failed_batch_retained_but_does_not_block_backlog_forever(tmp_path):
    state = journal(tmp_path)
    state.enqueue("chat", [str(i) for i in range(200)], 0)
    first = state.next_batch("chat", 0)
    assert not state.fail(first["id"], 0)
    assert state.next_batch("chat", 299) is None
    assert state.next_batch("chat", 300)["id"] == first["id"]
    assert not state.fail(first["id"], 300)
    assert state.fail(first["id"], 600)
    second = state.next_batch("chat", 600)
    assert second["id"] != first["id"]
    with state.connect() as db:
        failed = db.execute("SELECT done,attempts FROM batches WHERE id=?", (first["id"],)).fetchone()
        assert tuple(failed) == (2, 3)


@pytest.mark.asyncio
async def test_jargon_inference_keeps_valid_evidence_for_replay(monkeypatch):
    from src.learners import jargon_miner as module
    from src.common.database.database_model import JargonCreatedBy

    miner = module.JargonMiner.__new__(module.JargonMiner)
    miner.session_id = "chat"
    refs = json.dumps([[{"platform": "qq", "message_id": "m"}]])
    item = NS(content="term", created_by=JargonCreatedBy.AI, count=4, meaning="", evidence_messages=refs)
    monkeypatch.setattr(miner, "_load_evidence_contexts", lambda item: (["real context"], refs, False))
    cleanup = AsyncMock()
    # The synchronous cleanup must not be invoked for unchanged, valid references.
    monkeypatch.setattr(miner, "_cleanup_jargon_evidence_messages", cleanup)
    monkeypatch.setattr(miner, "_modify_jargon_entry", lambda item: None)
    monkeypatch.setattr(miner, "_log_inference_prompt_preview", lambda **kw: None)
    monkeypatch.setattr(module.prompt_manager, "get_prompt", lambda name: NS(add_context=lambda *args: None))
    monkeypatch.setattr(module.prompt_manager, "render_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(
        module.llm_inference, "generate_response", AsyncMock(return_value=NS(response='{"no_info":true,"meaning":""}'))
    )
    await miner.infer_meaning(item)
    assert item.evidence_messages == refs
    cleanup.assert_not_called()
    miner._update_jargon(item, [[{"platform": "qq", "message_id": "m"}]], strict_persistence=True)
    assert item.count == 4
