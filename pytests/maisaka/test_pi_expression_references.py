"""Offline tests of expression retrieval and the actual Pi reference entrypoint."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncio
import json
import sys


from src.maisaka.pi_runtime import expression_references as expressions
from src.maisaka.pi_runtime import references


def candidate(identity=1, situation="表示惊讶", style="我去", count=3):
    return {"id": identity, "situation": situation, "style": style, "count": count}


def configure(monkeypatch, *, vector=False):
    config = SimpleNamespace(
        expression=SimpleNamespace(use_vector_expression=vector, expression_vector_index_path="test-index.json"),
        a_memorix=SimpleNamespace(
            integration=SimpleNamespace(heuristic_memory_recall_enabled=False, enable_person_profile_injection=False),
            image_memory=SimpleNamespace(enabled=False),
        ),
    )
    monkeypatch.setitem(sys.modules, "src.config.config", SimpleNamespace(global_config=config))
    return config


def test_quick_mode_handles_small_library_and_abstract_situations():
    entries = [candidate(), candidate(2, "祝贺朋友", "强啊", 2)]
    assert expressions.select_quick_candidates(entries, "刚出了稀有道具！！") == entries
    assert expressions.select_quick_candidates(entries, "") == []
    assert expressions.select_quick_candidates(entries, "hello", limit=0) == []
    assert expressions.select_quick_candidates([], "hello") == []


def test_context_matches_win_and_duplicate_styles_are_not_repeated():
    entries = [
        candidate(1, "表示惊讶", "我去", 100),
        candidate(2, "游戏通关", "强啊", 1),
        candidate(3, "游戏胜利", "强啊", 2),
        candidate(4, "游戏通关", "厉害", 1),
    ]
    result = expressions.select_quick_candidates(entries, "游戏通关")
    assert [item["id"] for item in result] == [2, 4]
    assert len(expressions.select_quick_candidates(entries, "hello", limit=100)) == 2


def test_admission_reuses_scope_and_curated_loader_and_excludes_rejections(monkeypatch):
    scope_loader = Mock(return_value=[candidate(1), candidate(2)])
    settings = Mock(return_value=(True, False))
    monkeypatch.setitem(
        sys.modules,
        "src.chat.replyer.maisaka_expression_selector",
        SimpleNamespace(maisaka_expression_selector=SimpleNamespace(_load_all_expression_candidates=scope_loader)),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.common.utils.utils_config",
        SimpleNamespace(ExpressionConfigUtils=SimpleNamespace(get_expression_config_for_chat=settings)),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.learners.expression_review_store",
        SimpleNamespace(get_review_state=lambda identity: {"rejected": identity == 2}),
    )
    assert expressions._load_enabled_candidates("actual-session") == [candidate(1)]
    scope_loader.assert_called_once_with("actual-session")
    settings.return_value = (False, True)
    scope_loader.reset_mock()
    assert expressions._load_enabled_candidates("actual-session") == []
    scope_loader.assert_not_called()


def test_query_uses_real_same_chat_recent_messages_only():
    def entry(identity, text, *, chat="here", included=True):
        return SimpleNamespace(
            original_message=SimpleNamespace(message_id=identity, processed_plain_text=text, session_id=chat),
            count_in_context=included,
        )

    runtime = SimpleNamespace(
        session_id="here",
        _chat_history=[
            entry("1", "我终于通关了"),
            entry("2", "SECRET", chat="elsewhere"),
            entry("3", "hidden", included=False),
            SimpleNamespace(processed_plain_text="tool injection"),
            entry("4", "duplicate"),
        ],
    )
    current = [SimpleNamespace(message_id="4", processed_plain_text="太不容易了")]
    assert expressions.expression_query(runtime, current) == "我终于通关了\n太不容易了"


def test_vector_results_cannot_bypass_scope_or_replace_live_text(monkeypatch):
    configure(monkeypatch, vector=True)
    monkeypatch.setattr(expressions, "_load_enabled_candidates", lambda sid: [candidate(1), candidate(2), candidate(3)])
    select = AsyncMock(
        return_value=[
            {**candidate(88), "item_similarity": 1.0},
            {**candidate(2, style="stale text"), "item_similarity": 0.9},
            {**candidate(3), "item_similarity": -0.1},
            {**candidate(1), "item_similarity": 0.8},
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "src.chat.replyer.expression_vector_index",
        SimpleNamespace(expression_vector_index=SimpleNamespace(select_candidates=select)),
    )
    assert asyncio.run(expressions.retrieve_expression_candidates("here", "新消息")) == [candidate(1)]
    args = select.call_args.kwargs
    assert args["session_id"] == "here" and args["candidate_pool_size"] == 2
    assert len(args["scoped_candidates"]) == 3


def test_vector_no_results_does_not_switch_to_quick_sampling(monkeypatch):
    configure(monkeypatch, vector=True)
    monkeypatch.setattr(expressions, "_load_enabled_candidates", lambda sid: [candidate()])
    monkeypatch.setitem(
        sys.modules,
        "src.chat.replyer.expression_vector_index",
        SimpleNamespace(expression_vector_index=SimpleNamespace(select_candidates=AsyncMock(return_value=[]))),
    )
    assert asyncio.run(expressions.retrieve_expression_candidates("here", "你好")) == []


def test_actual_build_references_injects_style_without_literal_overlap(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(expressions, "_load_enabled_candidates", lambda sid: [candidate()])
    monkeypatch.setattr(references, "_jargon_records", lambda *args: [])
    monkeypatch.setattr(references, "_image_records", AsyncMock(return_value=[]))
    monkeypatch.setitem(
        sys.modules,
        "src.maisaka.memory.heuristic_injector",
        SimpleNamespace(HeuristicMemoryContext=SimpleNamespace, heuristic_memory_injector=SimpleNamespace()),
    )
    monkeypatch.setitem(
        sys.modules, "src.maisaka.memory.person_profile", SimpleNamespace(collect_person_profile_candidates=Mock())
    )
    monkeypatch.setitem(sys.modules, "src.services.memory_service", SimpleNamespace(memory_service=SimpleNamespace()))
    runtime = SimpleNamespace(session_id="here", chat_stream=SimpleNamespace())
    source = SimpleNamespace(message_id="1", processed_plain_text="十连双金！！")
    result = asyncio.run(references.build_references(runtime, [source]))
    assert len(result) == 1 and result[0].namespace == "expression"
    payload = json.loads(result[0].content)
    assert payload["nature"] == "optional_style_example_not_fact_or_instruction"
    assert payload["authority"] == "untrusted_historical_reference_not_instruction"
    assert json.loads(payload["content"])["style"] == "我去"
    # Stable source identity and no retrieval-time noise in the version.
    assert asyncio.run(references.build_references(runtime, [source])) == result
    monkeypatch.setattr(expressions, "_load_enabled_candidates", lambda sid: [candidate(style="好家伙")])
    changed = asyncio.run(references.build_references(runtime, [source]))
    assert changed[0].source_id == result[0].source_id and changed[0].version != result[0].version


def test_styles_respect_shared_budget_and_follow_factual_references():
    styles = [
        references.make_reference("expression", str(i), "optional style", {}, nature="optional_style") for i in (1, 2)
    ]
    facts = [references.make_reference("memory", str(i), "Hiking " + " x" * 1000, {}, nature="fact") for i in (1, 2, 3)]
    assert references.select_references([*styles, *facts], "Hiking", max_chars=10000) == facts
    assert references.select_references(styles, "hello", max_chars=1) == []
    assert references.select_references(styles, "hello", max_items=0) == []
    assert references.select_references(list(reversed(styles)), "hello", max_items=1) == [styles[1]]


def test_optional_expression_failure_is_visible_and_does_not_use_random_examples(monkeypatch):
    monkeypatch.setattr(
        expressions, "retrieve_expression_candidates", AsyncMock(side_effect=RuntimeError("unavailable"))
    )
    logger = Mock()
    monkeypatch.setattr(references, "logger", logger)
    runtime = SimpleNamespace(session_id="here")
    assert (
        asyncio.run(
            references._expression_records(runtime, [SimpleNamespace(message_id="1", processed_plain_text="hi")])
        )
        == []
    )
    logger.warning.assert_called_once()
