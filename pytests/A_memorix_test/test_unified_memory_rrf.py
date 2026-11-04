from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.A_memorix.core.retrieval import (
    DualPathRetriever, DualPathRetrieverConfig, RetrievalResult,
    SparseBM25Config, VectorPoolsConfig,
)
from src.A_memorix.core.retrieval.threshold import DynamicThresholdFilter, ThresholdConfig
from src.A_memorix.core.runtime.models import KernelSearchRequest
from src.A_memorix.core.runtime.sdk_memory_kernel import SDKMemoryKernel
from src.A_memorix.core.storage import MetadataStore
from src.A_memorix.core.utils.search_execution_service import SearchExecutionService


class _Vectors:
    dimension = 4

    def __init__(self, ids, scores):
        self.ids, self.scores = ids, scores

    def search(self, query, k=10, **kwargs):
        allowed = kwargs.get("allowed_ids")
        pairs = [(h, s) for h, s in zip(self.ids, self.scores, strict=True) if allowed is None or h in allowed][:k]
        return [h for h, _ in pairs], [s for _, s in pairs]


class _Embedding:
    async def encode(self, query):
        return np.ones(4, dtype=np.float32)


class _Sparse:
    def __init__(self, ids):
        self.ids = ids
        self.calls = 0

    def search(self, query, k=10, allowed_ids=None):
        self.calls += 1
        return [
            {"hash": h, "score": 20.0 - rank, "bm25_score": -20.0 + rank,
             "matched_token_count": 2, "matched_token_ratio": 0.5}
            for rank, h in enumerate(self.ids) if allowed_ids is None or h in allowed_ids
        ][:k]


@pytest.mark.asyncio
async def test_explicit_hybrid_runs_keywords_despite_high_semantic_score(tmp_path):
    store = MetadataStore(tmp_path / "memory.db")
    store.connect()
    unrelated = [store.add_paragraph(f"马斯克在第{i}次游戏闲聊中接话") for i in range(12)]
    alias = store.add_paragraph("本群称爆!为顺之利刃，是让群友跪下的玩笑称号。")
    ids = unrelated[:8] + [alias] + unrelated[8:]
    vectors = _Vectors(ids, [0.9 - i * 0.005 for i in range(len(ids))])
    sparse = _Sparse([alias])
    retriever = DualPathRetriever(
        vectors, None, store, _Embedding(), sparse_index=sparse,
        graph_vector_store=_Vectors([], []),
        config=DualPathRetrieverConfig(
            enable_ppr=False, sparse=SparseBM25Config(mode="auto"),
            vector_pools=VectorPoolsConfig(mode="dual"),
        ),
    )
    old = await retriever.retrieve("马斯克 顺之利刃 称号", top_k=6)
    assert sparse.calls == 0
    assert alias not in [item.hash_value for item in old]
    new = await retriever.retrieve("马斯克 顺之利刃 称号", top_k=6, hybrid_rrf=True)
    assert sparse.calls == 1
    assert new[0].hash_value == alias
    assert new[0].metadata["keyword_match"] is True
    assert new[0].metadata["score_breakdown"]["score_calibration_method"] == "weighted_rrf"
    store.close()


def test_rrf_rank_does_not_turn_low_similarity_into_evidence():
    results = [
        RetrievalResult("noise", "无关内容", 0.99, "paragraph", "dual_vector_pool",
                        {"score_breakdown": {"semantic": 0.05}}),
        RetrievalResult("exact", "顺之利刃的记忆", 0.10, "paragraph", "dual_vector_pool",
                        {"keyword_match": True, "score_breakdown": {"semantic": 0.04}}),
    ]
    threshold = DynamicThresholdFilter(ThresholdConfig(min_results=4))
    assert [item.hash_value for item in SearchExecutionService._filter_hybrid_evidence(results, threshold)] == ["exact"]
    assert SearchExecutionService._filter_hybrid_evidence(results[:1], threshold) == []


class _StaticRetriever:
    config = SimpleNamespace(enable_ppr=False)

    def __init__(self, hits):
        self.hits = hits
        self.hybrid_calls = []

    async def retrieve(self, **kwargs):
        self.hybrid_calls.append(kwargs.get("hybrid_rrf"))
        return self.hits

    _is_temporal_match = DualPathRetriever._is_temporal_match
    def _matches_person_filter(self, *args):
        return True
    _extract_effective_time = DualPathRetriever._extract_effective_time


def _kernel(tmp_path, store, hits):
    kernel = SDKMemoryKernel(plugin_root=tmp_path, config={"retrieval": {"enable_ppr": False}})

    async def initialize():
        return None

    kernel.initialize = initialize
    kernel._initialized = True
    kernel.metadata_store = store
    kernel.retriever = _StaticRetriever(hits)
    kernel.graph_store = None
    kernel.vector_store = object()
    kernel.embedding_manager = object()
    kernel.episode_retriever = object()
    kernel.aggregate_query_service = object()
    kernel.threshold_filter = None
    return kernel


@pytest.mark.asyncio
async def test_search_unifies_fact_paragraph_entity_evidence_and_episode_with_scope(tmp_path):
    store = MetadataStore(tmp_path / "memory.db")
    store.connect()
    current = store.add_paragraph(
        "爆!沿用了顺之利刃这个玩笑称号", source="chat_summary:current",
        metadata={"chat_id": "current", "source_type": "chat_summary"},
    )
    fact = store.add_paragraph(
        "目标人物自称顺之利刃", source="person_fact:alice",
        metadata={"chat_id": "current", "person_ids": ["alice"], "source_type": "person_fact"},
    )
    other = store.add_paragraph(
        "顺之利刃的其他群秘密", source="chat_summary:other", metadata={"chat_id": "other"},
    )
    entity = store.add_entity("顺之利刃")
    store.link_paragraph_entity(current, entity)
    store.link_paragraph_entity(other, entity)
    store.upsert_episode({
        "episode_id": "current-episode", "source": "chat_summary:current",
        "title": "顺之利刃与跪下梗", "summary": "群里给机器人起了顺之利刃的玩笑称号。",
        "paragraph_count": 1, "participants": [], "keywords": ["顺之利刃"],
    })
    store.bind_episode_paragraphs("current-episode", [current])
    kernel = _kernel(tmp_path, store, [RetrievalResult(fact, "目标人物自称顺之利刃", .8, "paragraph", "test", {})])
    result = await kernel.search_memory(KernelSearchRequest(query="马斯克 顺之利刃 称号", limit=6, chat_id="current"))
    assert "error" not in result
    assert kernel.retriever.hybrid_calls == [True]
    assert {item.get("hash") for item in result["hits"]} >= {current, fact}
    assert any(item.get("episode_id") == "current-episode" for item in result["hits"])
    assert other not in [item.get("hash") for item in result["hits"]]
    assert len(result["hits"]) == 3
    store.close()


@pytest.mark.asyncio
async def test_entity_expansion_preserves_person_time_deleted_and_current_state_filters(tmp_path):
    store = MetadataStore(tmp_path / "memory")
    store.connect()
    entity = store.add_entity("顺之利刃")
    ids = []
    for label, person, event, metadata in [
        ("有效证据", "alice", 250, {}),
        ("其他人物", "bob", 250, {}),
        ("旧时间", "alice", 150, {}),
        ("已撤回", "alice", 250, {"memory_change": {"valid_to": 1.0}}),
        ("已删除", "alice", 250, {}),
    ]:
        h = store.add_paragraph(
            label + "关于顺之利刃的称号", source="person_fact:" + person,
            metadata={"chat_id": "current", "person_ids": [person], **metadata},
            time_meta={"event_time": event},
        )
        store.link_paragraph_entity(h, entity)
        ids.append(h)
    store.query("UPDATE paragraphs SET is_deleted=1 WHERE hash=?", (ids[-1],))
    store._conn.commit()
    kernel = _kernel(tmp_path, store, [])
    result = await kernel.search_memory(KernelSearchRequest(
        query="马斯克 顺之利刃 称号", mode="hybrid", limit=6, chat_id="current",
        person_id="alice", time_start=200.0, time_end=300.0,
    ))
    assert [item.get("hash") for item in result["hits"]] == [ids[0]]
    store.close()


def test_person_filter_accepts_the_stored_identity_without_a_named_entity(tmp_path):
    store = MetadataStore(tmp_path / "memory")
    store.connect()
    h = store.add_paragraph("自称顺之利刃", metadata={"person_ids": ["alice"]})
    retriever = DualPathRetriever(
        _Vectors([], []), None, store, _Embedding(),
        config=DualPathRetrieverConfig(enable_ppr=False),
    )
    assert retriever._matches_person_filter(h, "alice")
    assert not retriever._matches_person_filter(h, "bob")
    store.close()
