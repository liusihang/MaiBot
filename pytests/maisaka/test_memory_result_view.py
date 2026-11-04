import json

from src.maisaka.memory.result_view import memory_result_page, memory_result_preview, parse_memory_result


def _payload():
    return {
        "status": "matched",
        "query": "马斯克 顺之利刃 称号",
        "requested_scope": {"chat_id": "PRIVATE_SESSION"},
        "effective_scope": {"chat_id": "PRIVATE_SESSION"},
        "hits": [
            {
                "hash": "PRIVATE_HASH",
                "type": "paragraph",
                "score": 0.9,
                "source": "dual_vector_pool",
                "content": "爆!在群里沿用了这个玩笑称号。",
                "metadata": {
                    "source_type": "chat_summary",
                    "chat_name": "大果训练营",
                    "retrieval_evidence": {"passed": True},
                    "event_time": "2026-10-28 15:26:20",
                },
            },
            {
                "type": "episode",
                "title": "顺之利刃与跪下梗名单扩展",
                "content": "群友围绕称号讨论接梗名单。",
                "metadata": {"source_type": "chat_summary"},
            },
        ],
    }


def test_preview_only_has_readable_evidence_and_a_continuation_handle():
    text = memory_result_preview(_payload(), "r123", max_chars=2400)
    assert "爆!在群里" in text and "大果训练营" in text
    assert "2026-10-28 15:26:20" in text
    assert "顺之利刃与跪下梗名单扩展" in text
    assert "r123" in text
    for diagnostic in ("PRIVATE_SESSION", "PRIVATE_HASH", "dual_vector_pool", "retrieval_evidence", "score", "requested_scope"):
        assert diagnostic not in text


def test_keyword_matches_body_or_title_and_returns_the_complete_entry():
    raw = _payload()
    page = memory_result_page(raw, 0, 2000, "顺之利刃")
    assert page["matched"] is True
    assert "顺之利刃与跪下梗名单扩展" in page["text"]
    assert "群友围绕称号讨论接梗名单。" in page["text"]
    assert "爆!在群里" not in page["text"]
    assert "requested_scope" not in json.dumps(page, ensure_ascii=False)


def test_query_echo_and_diagnostic_metadata_are_not_keyword_matches():
    raw = _payload()
    raw["hits"] = raw["hits"][:1]
    page = memory_result_page(raw, 0, 2000, "顺之利刃")
    assert page["matched"] is False and page["text"] == ""
    for keyword in ("PRIVATE_HASH", "PRIVATE_SESSION", "retrieval_evidence", "0.9"):
        assert memory_result_page(raw, 0, 2000, keyword)["matched"] is False


def test_long_entry_can_be_continued_without_losing_original_body():
    raw = {"status": "matched", "hits": [{"type": "paragraph", "content": "甲" * 4100 + "END_EVIDENCE"}]}
    parts, offset = [], 0
    while True:
        page = memory_result_page(raw, offset, 500)
        assert len(page["text"]) <= 500
        parts.append(page["text"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert "甲" * 4100 + "END_EVIDENCE" in "".join(parts)
    found = memory_result_page(raw, 0, 500, "END_EVIDENCE")
    assert "END_EVIDENCE" in found["text"]


def test_only_memory_envelopes_are_projected_and_preview_stays_bounded():
    assert parse_memory_result("ordinary tool text") is None
    assert parse_memory_result('{"hits": "not-a-list"}') is None
    assert parse_memory_result(json.dumps(_payload())) == _payload()
    raw = {"status": "matched", "hits": [{"content": "记忆" * 5000, "metadata": {"chat_name": "群" * 5000}}]}
    assert len(memory_result_preview(raw, "r123", max_chars=2400)) <= 2400
