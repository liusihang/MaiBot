"""Intent-based group language retrieval: BM25 + cosine ranks fused with RRF.

No generation/selector model is used. The catalog is the scope/status allowlist;
vector storage is shared infrastructure, never an alternative source of records.
"""

from collections import Counter
from hashlib import sha256
from typing import Any, Dict, List, Optional, Sequence, Tuple

import asyncio
import json
import math
import re
import time

from src.common.logger import get_logger
from src.common.paths import get_instance_root
from src.services import emoji_retrieval as vector_cache

logger = get_logger("group_expression_search")

RRF_K = 60
MAX_RESULTS = 6
MAX_ROUTE_CANDIDATES = 50
MAX_RESULT_CHARS = 12000
MAX_DOCUMENT_CHARS = 2400
MAX_INDEX_ITEMS_PER_QUERY = 8
INDEX_SECONDS_PER_QUERY = 8.0
# Calibrated on the deployed Qwen3-VL text embedding space; re-evaluate on model changes.
MIN_SEMANTIC_SIMILARITY = 0.58
INDEX_REFRESH_SECONDS = 30.0


async def load_group_expression_catalog(session_id: str) -> List[Dict[str, Any]]:
    from src.services.group_expression_catalog import load_group_expression_catalog as load

    return await load(session_id)


def tokenize(text: str) -> List[str]:
    """Unicode CJK bigrams and Latin/number words, without a new tokenizer model.

    Single-character CJK segments remain searchable. Longer segments use bigrams
    rather than unigram overlap, which otherwise overmatches unrelated sentences.
    """
    tokens = []
    for part in re.findall(r"[\u3400-\u9fff]+|[^\W_]+", text.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part) and len(part) > 1:
            tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def semantic_document(record: Dict[str, Any]) -> str:
    """Bound every field so a long phrase cannot hide its meaning or evidence."""
    examples = record.get("examples", [])[:3]
    parts = [
        "phrase: " + str(record.get("phrase", ""))[:400],
        "meaning: " + str(record.get("meaning", ""))[:600],
        "situation: " + str(record.get("situation", ""))[:600],
        "examples: " + "\n".join(str(example.get("text", ""))[:220] for example in examples),
    ]
    return "\n".join(parts)[:MAX_DOCUMENT_CHARS]


def bm25_scores(documents: Sequence[str], query: str) -> List[float]:
    """Okapi BM25 with k1=1.2, b=.75 and positive Robertson IDF."""
    terms = [Counter(tokenize(document)) for document in documents]
    lengths = [sum(item.values()) for item in terms]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    if not average:
        return [0.0] * len(terms)
    scores = [0.0] * len(terms)
    for token in set(tokenize(query)):
        frequency = sum(token in item for item in terms)
        idf = math.log(1 + (len(terms) - frequency + 0.5) / (frequency + 0.5))
        for index, item in enumerate(terms):
            count = item[token]
            if count:
                scores[index] += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * lengths[index] / average))
    return scores


def _prepare(records: List[Dict[str, Any]], query: str) -> Tuple[List[str], List[float], List[str]]:
    documents = [semantic_document(record) for record in records]
    # Structural embedding labels are not user language and must not generate
    # lexical matches for generic queries such as "meaning" or "examples".
    lexical_documents = [
        "\n".join(
            [
                str(record.get("phrase", ""))[:400],
                str(record.get("meaning", ""))[:600],
                str(record.get("situation", ""))[:600],
                *(str(example.get("text", ""))[:220] for example in record.get("examples", [])[:3]),
            ]
        )
        for record in records
    ]
    lexical = bm25_scores(lexical_documents, query)
    # Interleave query-relevant entries with deterministic hash-order breadth.
    # Unlike registration-order first-N, repeated queries advance durable progress
    # and a first query can reach a late relevant entry in a large catalog.
    relevant = sorted(range(len(records)), key=lambda index: (-lexical[index], records[index]["id"]))
    breadth = sorted(range(len(records)), key=lambda index: sha256(records[index]["id"].encode()).hexdigest())
    order = list(dict.fromkeys(value for pair in zip(relevant, breadth, strict=True) for value in pair))
    return documents, lexical, list(dict.fromkeys(documents[index] for index in order))


def _bounded_candidate(record: Dict[str, Any]) -> Dict[str, Any]:
    """Keep authentic prefixes, mark truncation rather than invent shorter evidence."""
    candidate = dict(record)
    truncated = []
    for field, maximum in (("phrase", 400), ("meaning", 1000), ("situation", 1000)):
        value = candidate.get(field, "")
        if len(value) > maximum:
            candidate[field] = value[:maximum]
            truncated.append(field)
    for field in ("source_ids", "related_ids"):
        if len(candidate.get(field, [])) > 20:
            candidate[field] = candidate[field][:20]
            truncated.append(field)
    examples = candidate.get("examples", [])
    if len(examples) > 3:
        truncated.append("examples")
    candidate["examples"] = []
    for original in examples[:3]:
        example = dict(original)
        if len(example.get("text", "")) > 600:
            example["text"] = example["text"][:600]
            example["truncated"] = True
        candidate["examples"].append(example)
    if truncated:
        candidate["truncated_fields"] = truncated
    return candidate


def _literal_match(record: Dict[str, Any], query: str) -> bool:
    """Allow explicit source phrases when a semantic comparison is unavailable."""
    phrase = record["phrase"].strip(' “”—「」"')
    phrases = [phrase, *re.findall(r'[“「"]([^”」"]+)[”」"]', phrase)]
    for value in phrases:
        if len(value) < 2 or "…" in value or "..." in value:
            continue
        pattern = re.escape(value)
        if value.isascii():
            pattern = r"(?<![a-z0-9_])" + pattern + r"(?![a-z0-9_])"
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


def _rank(
    records: List[Dict[str, Any]],
    documents: List[str],
    lexical: List[float],
    vectors: Dict[str, List[float]],
    query_vector: Optional[List[float]],
    limit: int,
    *,
    query: str = "",
) -> List[Dict[str, Any]]:
    cosine = {}
    if query_vector is not None:
        for index, document in enumerate(documents):
            vector = vectors.get(document)
            if vector is None:
                continue
            if len(vector) != len(query_vector):
                raise ValueError("Language cached vector dimension does not match query")
            score = sum(a * b for a, b in zip(vector, query_vector, strict=True))
            if math.isfinite(score):
                cosine[index] = score
    # Apply eligibility to both routes: keyword overlap cannot rescue a measured
    # weak semantic match. Missing vectors permit only explicit phrase matches.
    eligible = {
        index
        for index, record in enumerate(records)
        if (cosine[index] >= MIN_SEMANTIC_SIMILARITY if index in cosine else _literal_match(record, query))
    }
    lexical_order = sorted(
        (i for i, score in enumerate(lexical) if score > 0 and i in eligible),
        key=lambda i: (-lexical[i], records[i]["id"]),
    )[:MAX_ROUTE_CANDIDATES]
    semantic_order = sorted((i for i in cosine if i in eligible), key=lambda i: (-cosine[i], records[i]["id"]))[
        :MAX_ROUTE_CANDIDATES
    ]
    lexical_ranks = {index: rank for rank, index in enumerate(lexical_order, 1)}
    semantic_ranks = {index: rank for rank, index in enumerate(semantic_order, 1)}
    fused = {
        index: sum(1 / (RRF_K + ranks[index]) for ranks in (lexical_ranks, semantic_ranks) if index in ranks)
        for index in lexical_ranks.keys() | semantic_ranks.keys()
    }
    result = []
    for index in sorted(fused, key=lambda i: (-fused[i], records[i]["id"]))[:limit]:
        result.append(
            {
                **_bounded_candidate(records[index]),
                "ranking": {
                    "bm25_rank": lexical_ranks.get(index),
                    "semantic_rank": semantic_ranks.get(index),
                    "bm25_score": lexical[index],
                    "cosine_score": cosine.get(index),
                    "rrf_score": fused[index],
                },
            }
        )
    return result


def _index() -> vector_cache.EmojiDescriptionIndex:
    return vector_cache.EmojiDescriptionIndex(get_instance_root() / "data/pi_harness/group_language_vectors.sqlite3")


def _client() -> Any:
    from src.services.embedding_service import EmbeddingServiceClient

    return EmbeddingServiceClient(task_name="embedding", request_type="group_language.query")


def _coverage(total: int, indexed: int, *, space: Optional[str] = None, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "mode": "empty"
        if not total
        else "lexical_only"
        if error or not indexed
        else "hybrid"
        if indexed == total
        else "partial_hybrid",
        "total": total,
        "indexed": indexed,
        "complete": indexed == total,
        "lexical_complete": True,
        "embedding_space": space,
        "embedding_error": error,
        "rrf_k": RRF_K,
        "semantic_similarity_floor": MIN_SEMANTIC_SIMILARITY,
    }


async def search_group_expressions(
    session_id: str,
    intent: str,
    context: str = "",
    limit: int = 6,
    *,
    client: Any = None,
    index: Optional[vector_cache.EmojiDescriptionIndex] = None,
) -> Dict[str, Any]:
    """Retrieve scoped optional style references, disclosing incomplete semantics.

    At most one query and eight missing-document embedding calls per request.
    Failures are explicit in coverage, not disguised as completed hybrid search.
    """
    if not session_id or not intent.strip() or len(intent) > 1000 or len(context) > 2000:
        raise ValueError("Require session_id, intent of 1–1000 characters, context at most 2000")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RESULTS:
        raise ValueError("limit must be an integer from 1 to 6")
    records = await load_group_expression_catalog(session_id)
    if not records:
        return {"candidates": [], "coverage": _coverage(0, 0)}
    query = intent.strip() + ("\n" + context.strip() if context.strip() else "")
    documents, lexical, ordered = await asyncio.to_thread(_prepare, records, query)
    vectors: Dict[str, List[float]] = {}
    query_vector = None
    space = None
    error = None
    try:
        client = client if client is not None else _client()
        index = index if index is not None else _index()
        probe = await asyncio.wait_for(client.embed_text(query, session_id=session_id), INDEX_SECONDS_PER_QUERY)
        query_vector = vector_cache.unit_vector(probe.embedding)
        space = vector_cache._space(probe)
        vectors = await asyncio.to_thread(index.read, space, ordered)
        vectors = await vector_cache._populate(
            ordered,
            vectors,
            space=space,
            session_id=session_id,
            client=client,
            index=index,
            max_items=MAX_INDEX_ITEMS_PER_QUERY,
            max_seconds=INDEX_SECONDS_PER_QUERY,
        )
    except Exception as exc:
        # Do not leak provider response bodies/credentials through tool output.
        # Never use possibly mismatched vectors after a route/validation failure.
        error = type(exc).__name__
        logger.warning("Group language semantic retrieval failed (%s); returning explicit lexical-only coverage", error)
        vectors, query_vector = {}, None
    # Embedding can outlast a review/privacy edit. Revalidate live scope and
    # canonical content version before exposing any previously loaded evidence.
    fresh = await load_group_expression_catalog(session_id)
    allowed = {record["id"]: record["version"] for record in fresh}
    original_versions = {record["id"]: record["version"] for record in records}
    catalog_changed = original_versions != allowed
    unchanged = [record for record in records if allowed.get(record["id"]) == record["version"]]
    stale_removed = len(records) - len(unchanged)
    if catalog_changed:
        records = unchanged
        documents, lexical, _ = await asyncio.to_thread(_prepare, records, query)
    ranked = await asyncio.to_thread(_rank, records, documents, lexical, vectors, query_vector, limit, query=query)
    indexed = sum(document in vectors for document in documents)
    # Cap serialized model-visible evidence as well as per-field text. Keep a
    # ranked prefix, never silently substitute lower-ranked smaller records.
    bounded = []
    for candidate in ranked:
        if len(json.dumps([*bounded, candidate], ensure_ascii=False)) > MAX_RESULT_CHARS:
            break
        bounded.append(candidate)
    coverage = _coverage(len(fresh), indexed, space=space, error=error)
    coverage.update(
        {
            "catalog_changed": catalog_changed,
            "stale_records_removed": stale_removed,
            "lexical_complete": len(records) == len(fresh),
        }
    )
    coverage.update({"results_truncated": len(bounded) < len(ranked), "result_character_budget": MAX_RESULT_CHARS})
    return {"candidates": bounded, "coverage": coverage}


async def warm_group_expression_index(
    session_id: str,
    *,
    max_items: int = 100,
    max_seconds: float = 60.0,
    client: Any = None,
    index: Optional[vector_cache.EmojiDescriptionIndex] = None,
) -> Dict[str, Any]:
    """Explicit bounded maintenance API. Repeat until complete; no hidden paid job."""
    if not session_id or not 1 <= max_items <= 1000 or not math.isfinite(max_seconds) or not 0 < max_seconds <= 300:
        raise ValueError("Require session and budgets: 1–1000 items, 0–300 seconds")
    records = await load_group_expression_catalog(session_id)
    if not records:
        return _coverage(0, 0)
    documents, _, ordered = await asyncio.to_thread(_prepare, records, "")
    client = client if client is not None else _client()
    index = index if index is not None else _index()
    started = time.monotonic()
    probe = await asyncio.wait_for(
        client.embed_text("group language index", session_id=session_id), min(max_seconds, INDEX_SECONDS_PER_QUERY)
    )
    vector_cache.unit_vector(probe.embedding)
    space = vector_cache._space(probe)
    vectors = await asyncio.to_thread(index.read, space, ordered)
    vectors = await vector_cache._populate(
        ordered,
        vectors,
        space=space,
        session_id=session_id,
        client=client,
        index=index,
        max_items=max_items,
        max_seconds=max(0.0, max_seconds - (time.monotonic() - started)),
    )
    return _coverage(len(records), sum(document in vectors for document in documents), space=space)


class GroupLanguageIndexMaintainer:
    """Warm changed catalogs off the chat path; unchanged checks make no model calls."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.version: Optional[str] = None
        self.task: Optional[asyncio.Task] = None

    @staticmethod
    def signature(records: List[Dict[str, Any]]) -> str:
        return sha256(json.dumps(sorted((r["id"], r["version"]) for r in records)).encode()).hexdigest()

    async def sync_once(self) -> Optional[Dict[str, Any]]:
        records = await load_group_expression_catalog(self.session_id)
        version = self.signature(records)
        if version == self.version:
            return None
        if not records:
            self.version = version
            return _coverage(0, 0)
        progress = await warm_group_expression_index(self.session_id, max_items=100, max_seconds=60)
        # An incomplete batch retries using persisted vectors. Edits during a
        # warmup remain pending, even if the old snapshot reported completion.
        if progress["complete"] and version == self.signature(await load_group_expression_catalog(self.session_id)):
            self.version = version
            logger.info("Group language index synchronized: session=%s records=%d", self.session_id, len(records))
        return progress

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run(), name=f"group-language-index:{self.session_id}")

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def run(self) -> None:
        while True:
            try:
                await self.sync_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Group language index refresh failed (%s); retrying later", type(exc).__name__)
            await asyncio.sleep(INDEX_REFRESH_SECONDS)


def _language_session_ids() -> List[str]:
    """Discover actual stored sessions owning language records, without fabricating IDs."""
    from sqlmodel import select
    from src.common.database.database import get_db_session
    from src.common.database.database_model import ChatSession, Expression, Jargon

    with get_db_session(auto_commit=False) as db:
        known = set(db.exec(select(ChatSession.session_id)).all())
        owners = set(db.exec(select(Expression.session_id).distinct()).all())
        global_records = None in owners
        for scopes_json, is_global in db.exec(select(Jargon.session_id_dict, Jargon.is_global)).all():
            global_records = global_records or is_global
            try:
                scopes = json.loads(scopes_json)
            except (TypeError, ValueError):
                continue
            if isinstance(scopes, dict):
                owners.update(scopes)
    return sorted(known if global_records else known.intersection(owners))


async def maintain_known_group_language_indexes() -> None:
    """Start source-session workers at service startup, including quiet group chats."""
    workers: Dict[str, GroupLanguageIndexMaintainer] = {}
    try:
        while True:
            try:
                session_ids = set(await asyncio.to_thread(_language_session_ids))
                for session_id in session_ids:
                    if session_id not in workers:
                        workers[session_id] = GroupLanguageIndexMaintainer(session_id)
                        workers[session_id].start()
                for session_id in set(workers).difference(session_ids):
                    await workers.pop(session_id).stop()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Group language session discovery failed (%s); retrying later", type(exc).__name__)
            await asyncio.sleep(INDEX_REFRESH_SECONDS)
    finally:
        await asyncio.gather(*(worker.stop() for worker in workers.values()))
