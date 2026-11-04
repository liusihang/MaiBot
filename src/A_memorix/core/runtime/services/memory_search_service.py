from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import asyncio
import json

from src.A_memorix.core.retrieval import RetrievalScope, TemporalQueryOptions
from src.A_memorix.core.retrieval.score_calibration import fuse_score_maps
from src.A_memorix.core.utils.episode_retrieval_service import EpisodeRetrievalService
from src.A_memorix.core.utils.search_execution_service import (
    SearchExecutionRequest,
    SearchExecutionResult,
    SearchExecutionService,
)

from ..models import KernelSearchRequest, _NormalizedSearchTimeWindow
from .base import KernelServiceBase


class MemorySearchService(KernelServiceBase):
    """统一 SDK 检索入口，并在返回前收敛聊天范围和可见性规则。"""

    async def search_memory(self, request: KernelSearchRequest) -> Dict[str, Any]:
        """执行 search、time、hybrid、episode 或 aggregate 检索。

        共享聊天流只扩大候选召回范围，最终结果仍会经过人物、删除状态、聊天范围
        和检索类型过滤。普通检索交给 ``SearchExecutionService``，Episode 与聚合
        模式保留各自的执行器，最终统一转换为 SDK 命中结构。
        """
        if self._is_chat_filtered(
            respect_filter=request.respect_filter,
            stream_id=request.chat_id,
            group_id=request.group_id,
            user_id=request.user_id,
        ):
            return {"summary": "", "hits": [], "filtered": True, **self._search_status_fields()}

        await self.initialize()
        assert self.retriever is not None
        assert self.episode_retriever is not None
        assert self.aggregate_query_service is not None

        mode = str(request.mode or "search").strip().lower() or "search"
        query = str(request.query or "").strip()
        limit = max(1, int(request.limit or 5))
        shared_chat_ids = tuple(str(item or "").strip() for item in request.shared_chat_ids if str(item or "").strip())
        hit_service = self._kernel._get_search_hit_service()
        scope = type(hit_service)._resolve_retrieval_scope(
            hit_service, request.chat_id, shared_chat_ids
        )
        supported_modes = {"search", "time", "hybrid", "episode", "aggregate"}
        if mode not in supported_modes:
            return {
                "summary": "",
                "hits": [],
                "error": (f"不支持的检索模式: {mode}（仅支持 search/time/hybrid/episode/aggregate，semantic 已移除）"),
                **self._search_status_fields(),
            }
        try:
            time_window = self._normalize_search_time_window(request.time_start, request.time_end)
        except ValueError as exc:
            return {"summary": "", "hits": [], "error": str(exc), **self._search_status_fields()}

        if mode == "episode":
            rows = await self._episode_query_for_chat_scope(
                query=query,
                top_k=limit,
                time_from=time_window.numeric_start,
                time_to=time_window.numeric_end,
                person=request.person_id or None,
                scope=scope,
            )
            hits = self._filter_episode_hits([self._episode_hit(row) for row in rows])
            hits = self._filter_hits_by_retrieval_scope(hits, scope)
            if request.respect_filter:
                hits = self._filter_hits_by_retrieval_type_scope(
                    hits,
                    current_stream_id=request.chat_id,
                    current_group_id=request.group_id,
                    current_user_id=request.user_id,
                )
            hits = hits[:limit]
            return await self._finalize_search_response(hits)

        if mode == "aggregate":
            payload = await self.aggregate_query_service.execute(
                query=query,
                top_k=limit,
                mix=True,
                mix_top_k=limit,
                time_from=time_window.query_start,
                time_to=time_window.query_end,
                search_runner=lambda: self._aggregate_search(query, limit, request, scope),
                time_runner=lambda: self._aggregate_time(query, limit, request, time_window, scope),
                episode_runner=lambda: self._aggregate_episode(query, limit, request, time_window, scope),
            )
            hits = [dict(item) for item in payload.get("mixed_results", []) if isinstance(item, dict)]
            for item in hits:
                item.setdefault("metadata", {})
            filtered = self._filter_hits(hits, request.person_id)
            filtered = self._filter_user_visible_hits(filtered)
            filtered = self._filter_hits_by_retrieval_scope(filtered, scope)
            if request.respect_filter:
                filtered = self._filter_hits_by_retrieval_type_scope(
                    filtered,
                    current_stream_id=request.chat_id,
                    current_group_id=request.group_id,
                    current_user_id=request.user_id,
                )
            filtered = filtered[:limit]
            return await self._finalize_search_response(filtered)

        query_type = mode
        runtime_config = self._build_runtime_config()
        result = await self._search_execution_for_chat_scope(
            caller="sdk_memory_kernel",
            query_type=query_type,
            query=query,
            top_k=limit,
            request=request,
            time_from=time_window.query_start,
            time_to=time_window.query_end,
            plugin_config=runtime_config,
            scope=scope,
            enforce_chat_filter=bool(request.respect_filter),
        )
        if not result.success:
            return {"summary": "", "hits": [], "error": result.error, **self._search_status_fields()}
        if result.chat_filtered:
            return {"summary": "", "hits": [], "filtered": True, **self._search_status_fields()}

        hits = [self._retrieval_result_hit(item) for item in result.results]
        if mode in {"search", "hybrid"}:
            # 事实使用已投影的有效段落进入；实体用于定位支撑正文，避免重复返回空壳。
            entity_hits = await asyncio.to_thread(
                type(self)._entity_evidence_hits, self, query, request, time_window, scope
            )
            filtered = type(self)._visible_search_hits(self, hits + entity_hits, request, scope)
            episodes = await asyncio.to_thread(
                type(self)._related_episode_hits, self, query, filtered, request, time_window, scope
            )
            episode_hits = type(self)._visible_search_hits(self, episodes, request, scope)
            visible_ids = {type(self)._hit_identity(item) for item in filtered}
            # 实体投影补齐候选，不把同一段落的实体链接当作第二份独立检索证据。
            paragraphs = {}
            for item in hits + entity_hits:
                identity = type(self)._hit_identity(item)
                if identity in visible_ids:
                    paragraphs.setdefault(identity, item)
            branches = [list(paragraphs.values()), episode_hits]
            # 从过滤后的记录回填正文，不能把已失效的原始事实重新带回结果。
            visible = {type(self)._hit_identity(item): item for item in filtered + episode_hits}
            ranked = type(self)._rank_unified_hits(branches)
            filtered = [visible[type(self)._hit_identity(item)] for item in ranked]
        else:
            filtered = type(self)._visible_search_hits(self, hits, request, scope)
        return await self._finalize_search_response(filtered[:limit])

    @staticmethod
    def _hit_identity(hit: Dict[str, Any]) -> str:
        return str(hit.get("type", "")) + ":" + str(hit.get("episode_id") or hit.get("hash") or "")

    def _visible_search_hits(
        self, hits: List[Dict[str, Any]], request: KernelSearchRequest, scope: Optional[RetrievalScope]
    ) -> List[Dict[str, Any]]:
        # 先从数据库投影正文和人物归属，再执行定向人物过滤。
        hits = self._filter_user_visible_hits(hits)
        hits = self._filter_hits(hits, request.person_id)
        hits = self._filter_hits_by_retrieval_scope(hits, scope)
        if request.respect_filter:
            hits = self._filter_hits_by_retrieval_type_scope(
                hits, current_stream_id=request.chat_id,
                current_group_id=request.group_id, current_user_id=request.user_id,
            )
        return hits

    @staticmethod
    def _rank_unified_hits(branches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """按已通过证据过滤的通道名次融合不同记忆类型，不为相对分设置相关度门槛。"""
        records: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, Dict[str, float]] = {}
        for index, branch in enumerate(branches):
            channel = str(index)
            scores[channel] = {}
            for rank, hit in enumerate(branch, 1):
                identity = MemorySearchService._hit_identity(hit)
                records.setdefault(identity, hit)
                scores[channel].setdefault(identity, -float(rank))
        fused, _ = fuse_score_maps(scores, {channel: 1.0 for channel in scores}, method="weighted_rrf")
        return [records[key] for key in sorted(fused, key=lambda key: (-fused[key], key))]

    def _entity_evidence_hits(
        self, query: str, request: KernelSearchRequest,
        window: _NormalizedSearchTimeWindow, scope: Optional[RetrievalScope],
    ) -> List[Dict[str, Any]]:
        entities = self.metadata_store.query(
            "SELECT hash, name FROM entities WHERE COALESCE(is_deleted,0)=0 "
            "AND LENGTH(TRIM(name))>=2 AND INSTR(LOWER(?), LOWER(TRIM(name)))>0 "
            "AND (? IS NULL OR hash IN (SELECT value FROM json_each(?))) "
            "ORDER BY LENGTH(name) DESC, name ASC LIMIT 8",
            (query, None if scope is None else scope.key,
             json.dumps(sorted(scope.entity_ids)) if scope else "[]"),
        )
        if not entities:
            return []
        paragraphs = self.metadata_store.get_paragraphs_by_entity_hashes([row["hash"] for row in entities])
        temporal = TemporalQueryOptions(time_from=window.numeric_start, time_to=window.numeric_end)
        hits: Dict[str, Dict[str, Any]] = {}
        for entity in entities:
            for row in paragraphs.get(entity["hash"], []):
                if scope is not None and row["hash"] not in scope.paragraph_ids:
                    continue
                if (window.numeric_start is not None or window.numeric_end is not None) and not self.retriever._is_temporal_match(row, temporal):
                    continue
                metadata = dict(row.get("metadata") or {})
                metadata["retrieval_evidence"] = {"passed": True, "backfilled": False, "entity_match": entity["name"]}
                hits.setdefault(row["hash"], {
                    "hash": row["hash"], "type": "paragraph", "content": row["content"],
                    "source": row.get("source", ""), "metadata": metadata, "score": 1.0,
                })
                if len(hits) >= 80:
                    return list(hits.values())
        return list(hits.values())

    def _related_episode_hits(
        self, query: str, hits: List[Dict[str, Any]], request: KernelSearchRequest,
        window: _NormalizedSearchTimeWindow, scope: Optional[RetrievalScope],
    ) -> List[Dict[str, Any]]:
        bounds = {"time_from": window.numeric_start, "time_to": window.numeric_end,
                  "person": request.person_id or None}
        lexical = self.metadata_store.query_episodes(
            query=query, limit=max(20, request.limit * 4),
            allowed_episode_ids=scope.episode_ids if scope else None, **bounds,
        )
        paragraph_ids = [item["hash"] for item in hits if item.get("type") == "paragraph"]
        projected = self.metadata_store.get_episode_rows_by_paragraph_hashes(paragraph_ids, **bounds)
        if scope is not None:
            projected = [item for item in projected if item["episode_id"] in scope.episode_ids]
        evidence_ranks = {hash_value: rank for rank, hash_value in enumerate(paragraph_ids)}
        projected.sort(key=lambda item: min(
            (evidence_ranks.get(value, len(paragraph_ids)) for value in item.get("matched_paragraph_hashes", [])),
            default=len(paragraph_ids),
        ))
        # 复用 Episode 的词法与证据名次融合，不再为同一问题重复执行向量查询。
        rows = EpisodeRetrievalService(metadata_store=self.metadata_store)._fuse_branches(
            {"lexical": lexical, "paragraph_evidence": projected}, top_k=max(20, request.limit * 4)
        )
        return [self._episode_hit(row) for row in rows]

    async def _finalize_search_response(self, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """只为最终返回且实际采用的关系命中提交一次 ACCESS。"""

        relation_hashes: List[str] = []
        seen: set[str] = set()
        for item in hits:
            if str(item.get("type", "") or "").strip() != "relation":
                continue
            relation_hash = str(item.get("hash", "") or "").strip()
            if not relation_hash or relation_hash in seen:
                continue
            seen.add(relation_hash)
            relation_hashes.append(relation_hash)
        if relation_hashes:
            await self._runtime_facade.reinforce_access(relation_hashes)
        return {
            "summary": self._summary(hits),
            "hits": hits,
            **self._search_status_fields(),
        }

    def _search_status_fields(self) -> Dict[str, Any]:
        status = self._runtime_capability_status()
        return {
            "degraded": bool(status["degraded"]),
            "retrieval_ready": bool(status["retrieval_ready"]),
            "retrieval_mode": str(status["retrieval_mode"]),
            "available_channels": list(status["available_channels"]),
            "unavailable_channels": list(status["unavailable_channels"]),
        }

    async def _aggregate_search(
        self,
        query: str,
        limit: int,
        request: KernelSearchRequest,
        scope: Optional[RetrievalScope] = None,
    ) -> Dict[str, Any]:
        result = await self._search_execution_for_chat_scope(
            caller="sdk_memory_kernel.aggregate",
            query_type="search",
            query=query,
            top_k=limit,
            request=request,
            plugin_config=self._build_runtime_config(),
            enforce_chat_filter=False,
            scope=scope,
        )
        hits = [self._retrieval_result_hit(item) for item in result.results] if result.success else []
        hits = self._filter_hits_by_retrieval_scope(hits, scope)
        hits = hits[:limit]
        return {
            "success": result.success,
            "results": hits,
            "count": len(hits),
            "query_type": "search",
            "error": result.error,
        }

    async def _aggregate_time(
        self,
        query: str,
        limit: int,
        request: KernelSearchRequest,
        time_window: _NormalizedSearchTimeWindow,
        scope: Optional[RetrievalScope] = None,
    ) -> Dict[str, Any]:
        result = await self._search_execution_for_chat_scope(
            caller="sdk_memory_kernel.aggregate",
            query_type="time",
            query=query,
            top_k=limit,
            request=request,
            time_from=time_window.query_start,
            time_to=time_window.query_end,
            plugin_config=self._build_runtime_config(),
            enforce_chat_filter=False,
            scope=scope,
        )
        hits = [self._retrieval_result_hit(item) for item in result.results] if result.success else []
        hits = self._filter_hits_by_retrieval_scope(hits, scope)
        hits = hits[:limit]
        return {
            "success": result.success,
            "results": hits,
            "count": len(hits),
            "query_type": "time",
            "error": result.error,
        }

    async def _aggregate_episode(
        self,
        query: str,
        limit: int,
        request: KernelSearchRequest,
        time_window: _NormalizedSearchTimeWindow,
        scope: Optional[RetrievalScope] = None,
    ) -> Dict[str, Any]:
        assert self.episode_retriever
        rows = await self._episode_query_for_chat_scope(
            query=query,
            top_k=limit,
            time_from=time_window.numeric_start,
            time_to=time_window.numeric_end,
            person=request.person_id or None,
            scope=scope,
        )
        hits = self._filter_episode_hits([self._episode_hit(row) for row in rows])
        hits = self._filter_hits_by_retrieval_scope(hits, scope)
        return {"success": True, "results": hits, "count": len(hits), "query_type": "episode"}

    async def _search_execution_once(
        self,
        *,
        caller: str,
        query_type: str,
        query: str,
        top_k: int,
        request: KernelSearchRequest,
        plugin_config: dict,
        source: Optional[str],
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        enforce_chat_filter: bool,
        scope: Optional[RetrievalScope] = None,
    ) -> SearchExecutionResult:
        candidate_top_k = max(
            int(top_k),
            int(self._cfg("retrieval.top_k_paragraphs", 20))
            + int(self._cfg("retrieval.top_k_relations", 10)),
        )
        return await SearchExecutionService.execute(
            retriever=self.retriever,
            threshold_filter=self.threshold_filter,
            plugin_config=plugin_config,
            request=SearchExecutionRequest(
                caller=caller,
                stream_id=str(request.chat_id or "") or None,
                group_id=str(request.group_id or "") or None,
                user_id=str(request.user_id or "") or None,
                query_type=query_type,
                query=query,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
                time_from=time_from,
                scope=scope,
                time_to=time_to,
                person=str(request.person_id or "") or None,
                source=source,
                use_threshold=True,
                enable_ppr=bool(self._cfg("retrieval.enable_ppr", True)),
                hybrid_rrf=caller == "sdk_memory_kernel" and query_type in {"search", "hybrid"},
            ),
            enforce_chat_filter=enforce_chat_filter,
        )

    async def _search_execution_for_chat_scope(
        self,
        *,
        caller: str,
        query_type: str,
        query: str,
        top_k: int,
        request: KernelSearchRequest,
        plugin_config: dict,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        enforce_chat_filter: bool,
        scope: Optional[RetrievalScope] = None,
    ) -> SearchExecutionResult:
        if scope is None:
            service = self._kernel._get_search_hit_service()
            scope = type(service)._resolve_retrieval_scope(
                service, request.chat_id, request.shared_chat_ids
            )
        return await self._search_execution_once(
            caller=caller,
            query_type=query_type,
            query=query,
            top_k=top_k,
            request=request,
            plugin_config=plugin_config,
            source=None,
            time_from=time_from,
            time_to=time_to,
            enforce_chat_filter=enforce_chat_filter,
            scope=scope,
        )

    async def _episode_query_for_chat_scope(
        self,
        *,
        query: str,
        top_k: int,
        time_from: Optional[float],
        time_to: Optional[float],
        person: Optional[str],
        chat_id: str = "",
        shared_chat_ids: Sequence[str] = (),
        scope: Optional[RetrievalScope] = None,
    ) -> List[Any]:
        assert self.episode_retriever is not None
        if scope is None:
            service = self._kernel._get_search_hit_service()
            scope = type(service)._resolve_retrieval_scope(
                service, chat_id, shared_chat_ids
            )
        return await self.episode_retriever.query(
            query=query,
            top_k=top_k,
            time_from=time_from,
            time_to=time_to,
            person=person,
            source=None,
            scope=scope,
        )

    def _filter_hits_by_retrieval_scope(
        self,
        hits: List[Dict[str, Any]],
        scope: Optional[RetrievalScope],
    ) -> List[Dict[str, Any]]:
        service = self._kernel._get_search_hit_service()
        return type(service)._filter_hits_by_retrieval_scope(hits, scope)

    def _filter_hits_by_chat_scope(
        self,
        hits: List[Dict[str, Any]],
        chat_id: str,
        shared_chat_ids: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        service = self._kernel._get_search_hit_service()
        return type(service)._filter_hits_by_chat_scope(service, hits, chat_id, shared_chat_ids)
