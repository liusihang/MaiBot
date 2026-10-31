"""A small existing expression index is usable without ten learned examples."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio
import importlib.util
import sys

import numpy as np


def test_one_indexed_expression_is_recalled_with_real_vector_math(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "src.config.config", SimpleNamespace(config_manager=SimpleNamespace()))
    # Load the production module without importing src.chat's application startup.
    module_path = Path(__file__).parents[2] / "src/chat/replyer/expression_vector_index.py"
    spec = importlib.util.spec_from_file_location("test_expression_index_runtime", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    ExpressionVectorIndex = module.ExpressionVectorIndex
    ExpressionVectorIndexSnapshot = module.ExpressionVectorIndexSnapshot
    IndexedExpression = module.IndexedExpression
    expression_fingerprint = module.expression_fingerprint

    entry = {"id": 1, "situation": "表达惊讶", "style": "好家伙", "count": 2}
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    snapshot = ExpressionVectorIndexSnapshot(
        path=tmp_path / "index.json",
        mtime=0.0,
        embedding_model="test-model",
        expressions=[
            IndexedExpression(
                id=1,
                situation=entry["situation"],
                style=entry["style"],
                count=2,
                fingerprint=expression_fingerprint(1, entry["situation"], entry["style"]),
                embedding_profile_marker="profile",
                embedding_model="test-model",
                embedding_dimension=2,
                cluster_id=0,
                index=0,
            )
        ],
        profile_vectors={"profile": vectors},
        profile_cluster_centers={"profile": vectors},
    )
    index = ExpressionVectorIndex()
    monkeypatch.setattr(index, "_load_snapshot", lambda path: snapshot)
    monkeypatch.setattr(
        index, "get_current_embedding_profile", AsyncMock(return_value=SimpleNamespace(marker="profile", dimension=2))
    )
    monkeypatch.setattr(index, "_validate_embedding_result_profile", lambda *args, **kwargs: None)
    embedding = AsyncMock(return_value=SimpleNamespace(embedding=[1.0, 0.0]))
    monkeypatch.setitem(
        sys.modules,
        "src.services.embedding_service",
        SimpleNamespace(EmbeddingServiceClient=lambda **kwargs: SimpleNamespace(embed_text=embedding)),
    )

    async def recall(scoped):
        return await index.select_candidates(
            index_path=str(tmp_path / "index.json"),
            session_id="here",
            query_text="十连双金",
            scoped_candidates=scoped,
            candidate_pool_size=2,
            cluster_pool_size=16,
        )

    result = asyncio.run(recall([entry]))
    assert len(result) == 1 and result[0]["id"] == 1 and result[0]["item_similarity"] == 1.0
    embedding.assert_awaited_once()
    embedding.reset_mock()
    assert asyncio.run(recall([])) == []
    embedding.assert_not_awaited()
    # A stale index cannot supply a modified/deleted live record.
    assert asyncio.run(recall([{**entry, "style": "changed"}])) == []
    embedding.assert_not_awaited()
