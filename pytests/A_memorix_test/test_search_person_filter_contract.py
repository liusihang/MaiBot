"""人物过滤是范围约束，不得在无命中时返回其他人物的候选。"""

import pytest

from src.A_memorix.core.runtime.services.search_hit_processing_service import MemorySearchHitProcessingService


def test_person_filter_miss_remains_empty():
    hits = [{"content": "乙喜欢咖啡", "metadata": {"person_ids": ["person-b"]}}]
    assert MemorySearchHitProcessingService._filter_hits(hits, "person-a") == []
    assert MemorySearchHitProcessingService._filter_hits(hits, "") is hits


@pytest.mark.parametrize(
    "metadata",
    [
        {"person_ids": ["person-a"]},
        {"person_ids": "person-a"},
        {"participants": ["person-a"]},
    ],
)
def test_person_filter_keeps_exact_metadata_identity(metadata):
    hit = {"content": "甲的记忆", "metadata": metadata}
    assert MemorySearchHitProcessingService._filter_hits([hit], "person-a") == [hit]


@pytest.mark.parametrize("content", ["person-ab 的记忆", "other-person-a 的记忆", "person-a1 的记忆"])
def test_person_filter_does_not_accept_partial_identity(content):
    assert MemorySearchHitProcessingService._filter_hits([{"content": content, "metadata": {}}], "person-a") == []


def test_legacy_exact_identity_mention_is_still_searchable():
    # 兼容的是 ID 提及范围，不据此把正文中其他人的偏好归属给被提及者。
    hit = {"content": "提到person-a的一次谈话", "metadata": []}
    assert MemorySearchHitProcessingService._filter_hits([hit], "person-a") == [hit]
