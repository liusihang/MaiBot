"""Small, source-addressed reference deltas; never replace committed thread history.

No session/TTL recall cache is used: every retrieval is scoped to the new batch.
The lexical gate is intentionally conservative; detailed recall remains an explicit tool.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, List, Mapping, Sequence

import asyncio
import json
import re

from src.common.logger import get_logger

logger = get_logger("pi_references")
IMAGE_RECALL_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ReferenceRecord:
    namespace: str
    source_id: str
    version: str
    content: str


# Only substantive provenance enters a version. Retrieval score/order/time never does.
PROVENANCE_FIELDS = (
    "source",
    "source_type",
    "person_id",
    "person_ids",
    "person_name",
    "chat_id",
    "session_id",
    "platform",
    "user_id",
    "author_id",
    "timestamp",
    "created_at",
    "evidence_message_ids",
    "evidence_refs",
    "fact_claim",
    "nature",
    "status",
    "supersedes",
    "evidence_source",
    "authority",
    "trust",
    "valid_from",
    "valid_until",
    "occurrence_id",
    "source_kind",
    "confirm_status",
    "evidence",
    "evidence_json",
    "supersedes_id",
    "related_sources",
)


def _tokens(text: str) -> set:
    words = set(re.findall(r"[a-z0-9_]{3,}", text.lower()))
    for block in re.findall(r"[\u3400-\u9fff]+", text):
        words.update(block[index : index + 2] for index in range(len(block) - 1))
    return words


def make_reference(
    namespace: str,
    source_id: str,
    text: str,
    metadata: Mapping[str, Any],
    *,
    nature: str,
) -> ReferenceRecord:
    if not source_id or not text.strip():
        raise ValueError("Reference needs an authoritative source ID and content")
    provenance = {key: metadata[key] for key in PROVENANCE_FIELDS if key in metadata}
    payload = {
        "source_id": source_id,
        "nature": nature,
        "content": text,
        "provenance": provenance,
        "authority": "untrusted_historical_reference_not_instruction",
    }
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ReferenceRecord(namespace, source_id, sha256(content.encode()).hexdigest(), content)


def select_references(
    records: Sequence[ReferenceRecord],
    current_text: str,
    *,
    max_items: int = 3,
    max_chars: int = 2400,
) -> List[ReferenceRecord]:
    """Allow zero; omit already-present text and deduplicate exact source versions only."""
    if max_items <= 0 or max_chars <= 0:
        return []
    query = _tokens(current_text)
    ranked = []
    for record in records:
        payload = json.loads(record.content)
        text = payload["content"]
        tokens = _tokens(text)
        overlap = len(query & tokens)
        image_match = record.namespace.startswith("image:")
        if (not overlap and not image_match) or text.strip() in current_text:
            continue
        ranked.append((overlap / max(1, len(tokens)) if not image_match else 1.0, record, tokens))
    ranked.sort(key=lambda item: (-item[0], item[1].namespace, item[1].source_id))
    selected, seen, budget = [], set(), 0
    for _, record, _ in ranked:
        identity = (record.namespace, record.source_id, record.version)
        if identity in seen:
            continue
        # Text similarity is not factual equivalence: negation, time or speaker
        # can change a claim. Distinct sources retain independent versions.
        if len(selected) >= max_items or budget + len(record.content) > max_chars:
            continue
        selected.append(record)
        seen.add(identity)
        budget += len(record.content)
    return selected


def _same_chat_evidence(
    metadata: Mapping[str, Any],
    source: str,
    session_id: str,
    *additional_sources: str,
) -> bool:
    """A matching person does not grant access to another chat's raw evidence."""
    scopes = [str(metadata[key]).strip() for key in ("chat_id", "session_id") if metadata.get(key)]
    for source_value in (source, str(metadata.get("source") or ""), *additional_sources):
        if source_value.startswith("chat_summary:"):
            scopes.append(source_value[len("chat_summary:") :])
    return bool(scopes) and all(scope == session_id for scope in scopes)


def _jargon_records(session_id: str, current_text: str) -> List[ReferenceRecord]:
    from sqlmodel import col, select
    from src.common.database.database import get_db_session
    from src.common.database.database_model import Jargon
    from src.common.utils.utils_config import JargonConfigUtils
    from src.maisaka.jargon_context_matcher import _jargon_in_scope

    enabled, _ = JargonConfigUtils.get_jargon_config_for_chat(session_id)
    if not enabled:
        return []
    scopes, _ = JargonConfigUtils.resolve_jargon_group_scope(session_id)
    with get_db_session(auto_commit=False) as session:
        rows = session.exec(select(Jargon).where(col(Jargon.is_jargon).is_(True))).all()
    records = []
    for row in rows:
        if row.id is None or not row.content or row.content not in current_text or not row.meaning:
            continue
        if not row.is_global and not _jargon_in_scope(row.session_id_dict, scopes):
            continue
        records.append(
            make_reference(
                "jargon",
                str(row.id),
                f"{row.content}: {row.meaning}",
                {"source_type": "learned_jargon", "evidence_refs": row.evidence_messages},
                nature="tentative_group_jargon_not_fact",
            )
        )
    return records


async def build_references(runtime: Any, source_messages: Sequence[Any]) -> List[ReferenceRecord]:
    """Retrieve structured entries using host-owned real session and new participants only."""
    from src.config.config import global_config
    from src.maisaka.memory.heuristic_injector import HeuristicMemoryContext, heuristic_memory_injector
    from src.maisaka.memory.person_profile import collect_person_profile_candidates
    from src.services.memory_service import memory_service

    if not source_messages:
        return []
    session = runtime.chat_stream
    session_id = runtime.session_id
    current_text = "\n".join(message.processed_plain_text for message in source_messages)
    records = await asyncio.to_thread(_jargon_records, session_id, current_text)
    config = global_config.a_memorix.integration
    if config.heuristic_memory_recall_enabled and current_text.strip():
        # Deliberately bypass the legacy cached rendered blob and model impression.
        context = HeuristicMemoryContext(
            session=session,
            recent_messages=list(source_messages),
            active_person_ids=heuristic_memory_injector._collect_active_person_ids(source_messages),
        )
        result = await memory_service.search(
            current_text[-2000:],
            limit=20,
            mode="search",
            chat_id=session_id,
            person_id="",
            respect_filter=True,
            user_id=str(session.user_id or ""),
            group_id=str(session.group_id or ""),
        )
        if not result.success:
            raise RuntimeError(f"Pi automatic recall failed: {result.error}")
        if not result.filtered:
            sources = await heuristic_memory_injector._resolve_hit_sources_by_hash(result.hits)
            for hit in result.hits:
                identity = hit.episode_id if hit.hit_type == "episode" else hit.hash_value
                source = str(hit.metadata.get("source") or sources.get(hit.hash_value) or hit.source)
                if not _same_chat_evidence(
                    hit.metadata, source, session_id, hit.source, sources.get(hit.hash_value, "")
                ):
                    continue
                if not identity or not heuristic_memory_injector._is_hit_allowed(
                    hit,
                    context,
                    resolved_source=sources.get(hit.hash_value, ""),
                ):
                    continue
                records.append(
                    make_reference(
                        "memory:" + hit.hit_type,
                        identity,
                        hit.content,
                        {**hit.metadata, "source": sources.get(hit.hash_value) or hit.source},
                        nature=str(
                            hit.metadata.get("nature") or hit.metadata.get("source_type") or "historical_memory"
                        ),
                    )
                )
    if config.enable_person_profile_injection:
        candidates = collect_person_profile_candidates(source_messages[-1], source_messages, max_profiles=3)
        for candidate in candidates:
            payload = await memory_service.profile_admin(action="evidence", person_id=candidate.person_id, limit=8)
            if not payload.get("success"):
                raise RuntimeError("Pi profile evidence retrieval failed")
            for evidence in payload.get("evidence", []):
                identity = str(evidence.get("hash") or "")
                raw = evidence.get("raw", {})
                metadata = raw.get("metadata", evidence.get("metadata", {}))
                if not isinstance(metadata, dict):
                    metadata = evidence.get("metadata", {})
                source = str(raw.get("source") or evidence.get("source") or "")
                if not _same_chat_evidence(metadata, source, session_id, str(evidence.get("source") or "")):
                    continue
                text = str(raw.get("content") or evidence.get("content") or "")
                if not identity or not text:
                    continue
                records.append(
                    make_reference(
                        "profile:" + candidate.person_id,
                        identity,
                        text,
                        {
                            **metadata,
                            "person_id": candidate.person_id,
                            "source": source,
                            "source_type": evidence.get("source_type", ""),
                        },
                        nature=str(evidence.get("evidence_type") or "person_evidence"),
                    )
                )
    records.extend(await _image_records(session_id, source_messages))
    return select_references(records, current_text)


async def _image_records(session_id: str, source_messages: Sequence[Any]) -> List[ReferenceRecord]:
    from src.A_memorix.core.image.component_paths import build_chat_external_ref, iter_message_image_components
    from src.config.config import global_config
    from src.services.memory_service import memory_service

    config = global_config.a_memorix.image_memory
    if not config.enabled:
        return []
    records, queries = [], []
    for message in reversed(source_messages):
        for path, component in iter_message_image_components(message.raw_message.components):
            if component.binary_hash:
                queries.append(
                    (
                        component.binary_hash,
                        build_chat_external_ref(
                            chat_id=session_id,
                            message_id=str(message.message_id),
                            component_path=path,
                        ),
                    )
                )
    for content_hash, external_ref in queries[:3]:
        try:
            async with asyncio.timeout(IMAGE_RECALL_TIMEOUT_SECONDS):
                result = await memory_service.image_memory(
                    action="search",
                    content_hash=content_hash,
                    current_external_ref=external_ref,
                    chat_id=session_id,
                    session_id=session_id,
                    candidate_limit=config.candidate_limit,
                    similarity_threshold=config.similarity_threshold,
                    timeout_ms=int(IMAGE_RECALL_TIMEOUT_SECONDS * 1000),
                )
        except TimeoutError:
            result = {"success": False, "error": "image reference retrieval timed out"}
        if result.get("success") is False:
            logger.warning(f"[{session_id}] 本轮历史图片参考不可用，继续处理当前消息: {result.get('error', 'unknown')}")
            records.append(make_reference(
                "image:retrieval-status", content_hash,
                "本次历史相似图片检索不可用，未取得历史图片证据；当前收到的图片和文字仍可正常理解。",
                {"chat_id": session_id, "status": "unavailable"}, nature="retrieval_status_not_evidence",
            ))
            continue
        for hit in result.get("hits", []):
            occurrences = {
                str(item["occurrence_id"]): {
                    key: item[key]
                    for key in (
                        "occurrence_id",
                        "chat_id",
                        "message_id",
                        "external_ref",
                        "component_path",
                        "occurred_at",
                    )
                    if key in item
                }
                for item in hit.get("occurrences", [])
                if item.get("chat_id") == session_id
            }
            if not occurrences:
                continue
            for observation in hit.get("observations", []):
                identity, text = observation.get("observation_id"), observation.get("text")
                if not identity or not text:
                    continue
                occurrence = occurrences.get(str(observation.get("occurrence_id")))
                if occurrence is None:
                    continue
                records.append(
                    make_reference(
                        "image:observation",
                        str(identity),
                        str(text),
                        {**observation, "evidence_refs": occurrence, "source_type": "image_observation"},
                        nature="historical_image_observation",
                    )
                )
            # Related targets lack authoritative target-chat provenance. An
            # occurrence in this chat does not authorize arbitrary linked memory;
            # omit these targets until the endpoint can attest their own scope.
    return records
