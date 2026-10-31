"""Scoped expression retrieval for the single Agent, without a selector model.

Quick mode ranks local examples and lets the main Agent judge their applicability.
Vector mode reuses the existing expression index; it never restores the Replyer.
"""

from typing import Any, Dict, List, Sequence

import asyncio
import math
import re

MAX_EXPRESSION_REFERENCES = 2


def _terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9_]{2,}", text.casefold()))
    for block in re.findall(r"[\u3400-\u9fff]+", text):
        terms.update(block[index : index + 2] for index in range(len(block) - 1))
    return terms


def expression_query(runtime: Any, source_messages: Sequence[Any]) -> str:
    """Use only real recent conversation, not injected memories or tool output."""
    current_ids = {str(message.message_id) for message in source_messages}
    previous = []
    for entry in getattr(runtime, "_chat_history", ()):
        original = getattr(entry, "original_message", None)
        if original is None or not entry.count_in_context or str(original.message_id) in current_ids:
            continue
        if original.session_id != runtime.session_id:
            continue
        previous.append((original.processed_plain_text or "").strip())
    current = [(message.processed_plain_text or "").strip() for message in source_messages]
    return "\n".join([*previous[-8:], *current]).strip()[-2000:]


def _load_enabled_candidates(session_id: str) -> List[Dict[str, Any]]:
    from src.chat.replyer.maisaka_expression_selector import maisaka_expression_selector
    from src.common.utils.utils_config import ExpressionConfigUtils
    from src.learners.expression_review_store import get_review_state

    use, _ = ExpressionConfigUtils.get_expression_config_for_chat(session_id)
    if not use:
        return []
    # Reuse the current scope, global-sharing, curated-only and normalization rules.
    candidates = maisaka_expression_selector._load_all_expression_candidates(session_id)
    return [item for item in candidates if not get_review_state(item["id"])["rejected"]]


def select_quick_candidates(
    candidates: Sequence[Dict[str, Any]],
    query: str,
    *,
    limit: int = MAX_EXPRESSION_REFERENCES,
) -> List[Dict[str, Any]]:
    """Prefer contextual matches, then bounded common habits in explicit quick mode.

    A situation such as 'express surprise' need not literally occur in a message
    expressing surprise. Unmatched examples remain optional, never inferred facts.
    Stable ordering avoids introducing a new reference version on every turn.
    """
    if not query.strip() or limit <= 0:
        return []
    query_terms = _terms(query)
    ranked = []
    for item in candidates:
        situation, style = str(item["situation"]).strip(), str(item["style"]).strip()
        if not situation or not style:
            continue
        terms = _terms(situation + " " + style)
        overlap = len(terms & query_terms) / max(1, len(terms))
        frequency = math.log1p(max(0, int(item.get("count") or 0)))
        ranked.append((overlap, frequency, int(item["id"]), item))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    selected, styles = [], set()
    for _, _, _, item in ranked:
        style_key = " ".join(str(item["style"]).casefold().split())
        if style_key in styles:
            continue
        selected.append(item)
        styles.add(style_key)
        if len(selected) >= min(limit, MAX_EXPRESSION_REFERENCES):
            break
    return selected


async def retrieve_expression_candidates(session_id: str, query: str) -> List[Dict[str, Any]]:
    from src.config.config import global_config

    if not query.strip():
        return []
    candidates = await asyncio.to_thread(_load_enabled_candidates, session_id)
    if not candidates:
        return []
    config = global_config.expression
    if not config.use_vector_expression:
        return select_quick_candidates(candidates, query)

    from src.chat.replyer.expression_vector_index import expression_vector_index

    matches = await expression_vector_index.select_candidates(
        index_path=config.expression_vector_index_path,
        session_id=session_id,
        query_text=query,
        scoped_candidates=candidates,
        candidate_pool_size=MAX_EXPRESSION_REFERENCES,
        cluster_pool_size=16,
    )
    # Revalidate against the admitted DB snapshot. Index text/IDs are not authority.
    admitted = {item["id"]: item for item in candidates}
    selected, styles = [], set()
    for match in matches:
        item = admitted.get(match["id"])
        if item is None or item["situation"] != match["situation"] or item["style"] != match["style"]:
            continue
        style_key = " ".join(str(item["style"]).casefold().split())
        if style_key in styles or float(match.get("item_similarity", 0.0)) <= 0:
            continue
        selected.append(item)
        styles.add(style_key)
        if len(selected) >= MAX_EXPRESSION_REFERENCES:
            break
    return selected
