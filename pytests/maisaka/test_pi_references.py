import json

import pytest

from src.maisaka.pi_runtime.references import make_reference, select_references


def ref(key, text, **metadata):
    return make_reference("memory", key, text, metadata, nature="historical_request")


def test_source_versions_ignore_retrieval_noise_and_preserve_authority():
    a = ref("m1", "Alice asked for kneeling yesterday", score=0.4, query="one", evidence_message_ids=["42"])
    b = ref("m1", "Alice asked for kneeling yesterday", score=0.9, query="two", evidence_message_ids=["42"])
    assert a == b
    changed = ref("m1", "Alice withdrew her request", evidence_message_ids=["43"])
    assert a.source_id == changed.source_id and a.version != changed.version
    assert json.loads(a.content)["authority"] == "untrusted_historical_reference_not_instruction"
    assert json.loads(a.content)["provenance"]["evidence_message_ids"] == ["42"]


def test_small_relevant_batch_only_and_no_blob_identity():
    a = ref("m1", "Alice likes mountain hiking")
    b = ref("m2", "Bob likes playing chess")
    first = select_references([a], "Alice mountain trip")
    second = select_references([a, b], "Alice mountain trip")
    assert first == second == [a]
    assert select_references([a, b], "new topic about astronomy") == []
    assert select_references([a], "Alice likes mountain hiking") == []
    assert select_references([a], "Alice hiking", max_chars=10) == []


def test_duplicates_budget_and_chinese_matching():
    a = ref("m1", "小明喜欢周末爬山，去年登过泰山")
    b = ref("m2", "小明喜欢周末爬山，去年登过泰山。")
    selected = select_references([a, b], "小明周末去爬山吗")
    assert len(selected) == 2
    assert len(select_references([a, b], "小明周末去爬山吗", max_items=1)) == 1


def test_distinct_sources_and_negations_keep_independent_versions():
    a = ref("a", "Alice likes hiking mountains", evidence_message_ids=["original-1"])
    b = ref("b", "Alice does not like hiking mountains", evidence_message_ids=["original-2"])
    selected = select_references([a, a, b], "Alice hiking", max_chars=4000)
    assert len(selected) == 2
    assert set(selected) == {a, b}
    assert select_references([a], "Alice hiking", max_items=0) == []
    assert select_references([a, b], "Alice hiking", max_items=1) == [a]
    # Candidate-set changes never manufacture a new version of an existing source.
    assert select_references([a], "Alice hiking")[0] == a


def test_build_references_real_endpoint_shapes_and_strict_raw_scope(monkeypatch):
    from types import SimpleNamespace
    import asyncio
    import sys

    from src.maisaka.pi_runtime import references

    calls = []
    integration = SimpleNamespace(heuristic_memory_recall_enabled=True, enable_person_profile_injection=True)
    image_config = SimpleNamespace(enabled=True, candidate_limit=8, similarity_threshold=0.8)
    monkeypatch.setitem(
        sys.modules,
        "src.config.config",
        SimpleNamespace(
            global_config=SimpleNamespace(
                a_memorix=SimpleNamespace(integration=integration, image_memory=image_config),
            )
        ),
    )

    async def resolve(hits):
        return {}

    injector = SimpleNamespace(
        _collect_active_person_ids=lambda messages: {"person-1"},
        _resolve_hit_sources_by_hash=resolve,
        # Regression: the old person-only scope check is deliberately permissive.
        _is_hit_allowed=lambda *args, **kwargs: True,
    )
    monkeypatch.setitem(
        sys.modules,
        "src.maisaka.memory.heuristic_injector",
        SimpleNamespace(
            HeuristicMemoryContext=SimpleNamespace,
            heuristic_memory_injector=injector,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.maisaka.memory.person_profile",
        SimpleNamespace(
            collect_person_profile_candidates=lambda *args, **kwargs: [SimpleNamespace(person_id="person-1")],
        ),
    )
    monkeypatch.setattr(references, "_jargon_records", lambda *args: [])

    def hit(identity, chat, text):
        return SimpleNamespace(
            hash_value=identity,
            episode_id="",
            hit_type="paragraph",
            content=text,
            metadata={"chat_id": chat, "evidence_message_ids": [identity + "-msg"]},
            source="person_fact:person-1",
        )

    async def search(query, **kwargs):
        calls.append((query, kwargs))
        return SimpleNamespace(
            success=True,
            filtered=False,
            error="",
            hits=[
                hit("m1", "group-1", "Hiking safety needs water"),
                hit("m2", "other-group", "Hiking SECRET other group"),
                hit("m3", "", "Hiking SECRET unknown scope"),
                SimpleNamespace(
                    hash_value="m4",
                    episode_id="",
                    hit_type="paragraph",
                    content="Hiking SECRET conflicting session",
                    source="person_fact:person-1",
                    metadata={"chat_id": "group-1", "session_id": "other-group"},
                ),
                SimpleNamespace(
                    hash_value="m5",
                    episode_id="",
                    hit_type="paragraph",
                    content="Hiking SECRET conflicting summary",
                    source="chat_summary:other-group",
                    metadata={"chat_id": "group-1"},
                ),
            ],
        )

    async def profile_admin(**kwargs):
        assert kwargs["action"] == "evidence"
        return {
            "success": True,
            "evidence": [
                {
                    "hash": "p1",
                    "content": "trimmed",
                    "source_type": "person_fact",
                    "raw": {
                        "content": "Hiking mountain routes researched last winter",
                        "source": "person_fact:person-1",
                        "metadata": {"chat_id": "group-1", "evidence_message_ids": ["raw-original"]},
                    },
                },
                # Raw provenance wins over an outer summary with another scope.
                {
                    "hash": "p2",
                    "content": "Hiking SECRET cross profile",
                    "metadata": {"chat_id": "group-1"},
                    "raw": {"content": "Hiking SECRET cross profile", "metadata": {"chat_id": "other-group"}},
                },
            ],
        }

    async def image_memory(**kwargs):
        assert kwargs["session_id"] == kwargs["chat_id"] == "group-1"
        return {
            "success": True,
            "hits": [
                {
                    "occurrences": [
                        {
                            "occurrence_id": "o1",
                            "chat_id": "group-1",
                            "message_id": "image-original",
                            "updated_at": "dynamic",
                        },
                        {"occurrence_id": "o2", "chat_id": "other-group", "message_id": "secret"},
                    ],
                    "observations": [
                        {
                            "observation_id": "i1",
                            "occurrence_id": "o1",
                            "text": "Hiking image shows red boots",
                            "source_kind": "user",
                            "confirm_status": "confirmed",
                            "evidence_json": "{}",
                        },
                        {"observation_id": "i2", "occurrence_id": "o2", "text": "SECRET other image"},
                    ],
                    "related_memories": [
                        {"target_type": "paragraph", "target_id": "mixed", "content": "SECRET mixed link"}
                    ],
                },
                {
                    # Current-group occurrence does not authorize a linked foreign target.
                    "occurrences": [{"occurrence_id": "local", "chat_id": "group-1", "message_id": "local-image"}],
                    "observations": [],
                    "related_memories": [
                        {
                            "target_type": "paragraph",
                            "target_id": "foreign-target",
                            "content": "Hiking SECRET foreign memory linked to local image",
                        }
                    ],
                },
            ],
        }

    monkeypatch.setitem(
        sys.modules,
        "src.services.memory_service",
        SimpleNamespace(
            memory_service=SimpleNamespace(
                search=search,
                profile_admin=profile_admin,
                image_memory=image_memory,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.A_memorix.core.image.component_paths",
        SimpleNamespace(
            iter_message_image_components=lambda components: [("0", SimpleNamespace(binary_hash="image-hash"))],
            build_chat_external_ref=lambda **kwargs: "chat:group-1:new:0",
        ),
    )
    runtime = SimpleNamespace(session_id="group-1", chat_stream=SimpleNamespace(user_id="", group_id="1"))
    message = SimpleNamespace(
        processed_plain_text="Hiking weekend plan", message_id="new", raw_message=SimpleNamespace(components=[])
    )
    result = asyncio.run(references.build_references(runtime, [message]))
    assert {item.source_id for item in result} == {"m1", "p1", "i1"}
    assert calls[0][0] == "Hiking weekend plan"
    assert calls[0][1]["chat_id"] == "group-1"
    output = "".join(item.content for item in result)
    assert "SECRET" not in output and "dynamic" not in output
    assert "raw-original" in output and "image-original" in output and "m1-msg" in output
    assert "trimmed" not in output


def test_all_known_evidence_scopes_must_agree():
    from src.maisaka.pi_runtime.references import _same_chat_evidence

    assert _same_chat_evidence({"chat_id": "a", "session_id": "a"}, "chat_summary:a", "a")
    assert not _same_chat_evidence({"chat_id": "a", "session_id": "b"}, "", "a")
    assert not _same_chat_evidence({"chat_id": "a"}, "chat_summary:b", "a")
    assert not _same_chat_evidence({"chat_id": "a", "source": "chat_summary:b"}, "chat_summary:a", "a")
    assert not _same_chat_evidence({}, "person_fact:p", "a")
    assert not _same_chat_evidence({"chat_id": "a"}, "chat_summary:a", "a", "chat_summary:b")


@pytest.mark.parametrize("failure", ["unavailable", "timeout", "cancelled"])
def test_optional_image_recall_failure_keeps_current_chat_available(monkeypatch, failure):
    import asyncio
    import sys
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from src.maisaka.pi_runtime import references
    config = SimpleNamespace(enabled=True, candidate_limit=8, similarity_threshold=0.8)
    monkeypatch.setitem(sys.modules, "src.config.config", SimpleNamespace(global_config=SimpleNamespace(a_memorix=SimpleNamespace(image_memory=config))))
    api = AsyncMock(return_value={"success": False, "error": "image space changed"})
    if failure == "timeout":
        async def slow(**kwargs):
            await asyncio.sleep(1)
        api.side_effect = slow
        monkeypatch.setattr(references, "IMAGE_RECALL_TIMEOUT_SECONDS", 0.01)
    elif failure == "cancelled":
        api.side_effect = asyncio.CancelledError
    monkeypatch.setitem(sys.modules, "src.services.memory_service", SimpleNamespace(memory_service=SimpleNamespace(image_memory=api)))
    monkeypatch.setitem(sys.modules, "src.A_memorix.core.image.component_paths", SimpleNamespace(
        iter_message_image_components=lambda parts: [("0", SimpleNamespace(binary_hash="picture"))],
        build_chat_external_ref=lambda **kwargs: "current-image",
    ))
    message = SimpleNamespace(message_id="42", raw_message=SimpleNamespace(components=[]))
    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(references._image_records("group", [message]))
        return
    result = asyncio.run(references._image_records("group", [message]))
    assert len(result) == 1
    assert json.loads(result[0].content)["nature"] == "retrieval_status_not_evidence"
    assert "不可用" in json.loads(result[0].content)["content"]
