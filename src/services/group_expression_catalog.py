"""Live, scoped expression/jargon catalog with verifiable same-chat examples.

Sharing a term never shares the raw conversation behind it. Existing tables remain
independent; this adapter exposes one retrieval interface and never synthesizes evidence.
"""

from pathlib import Path
from typing import Any, Dict, List

import asyncio
import json

from sqlmodel import col, select

from src.chat.message_receive.message import SessionMessage
from src.chat.replyer.maisaka_expression_selector import maisaka_expression_selector
from src.chat.utils.utils import is_bot_self
from src.common.database.database import ROOT_PATH, get_db_session
from src.common.database.database_model import Expression, Jargon, Messages
from src.common.utils.utils_config import ExpressionConfigUtils, JargonConfigUtils
from src.learners.expression_provenance import ExpressionProvenance, expression_identity, text_digest
from src.learners.expression_review_store import get_review_state
from src.learners.expression_style_utils import normalize_expression_style_for_learning
from src.learners.learning_text import extract_learning_text
from src.maisaka.jargon_context_matcher import _jargon_in_scope

MAX_EXAMPLES = 3


def _eligible_text(row: Messages) -> str:
    if row.is_notify or row.is_command or is_bot_self(row.platform, row.user_id):
        return ""
    return extract_learning_text(SessionMessage.from_db_instance(row).raw_message)


def _example(row: Messages, text: str, *, is_source: bool) -> Dict[str, Any]:
    return {
        "message_id": row.message_id,
        "text": text[:600],
        "sender_id": row.user_id,
        "time": row.timestamp.isoformat(),
        "is_source": is_source,
        "truncated": len(text) > 600,
    }


def _resolve_examples(db: Any, session_id: str, refs: List[Dict[str, Any]], *, phrase: str = "") -> tuple:
    examples, source_ids, seen = [], [], set()

    def resolve(ref: Dict[str, Any], platform: str) -> Any:
        mid = ref.get("message_id")
        if not isinstance(mid, str) or not mid or not platform:
            return None
        rows = db.exec(
            select(Messages)
            .where(
                Messages.session_id == session_id,
                Messages.platform == platform,
                Messages.message_id == mid,
            )
            .limit(2)
        ).all()
        # Platform IDs can collide; ambiguous rows cannot be attributed safely.
        if len(rows) != 1:
            return None
        row = rows[0]
        text = _eligible_text(row)
        if not text or ("text_hash" in ref and ref["text_hash"] != text_digest(text)):
            return None
        if "example" in ref and ref["example"] != text[:600]:
            return None
        return row, text

    for ref in refs:
        if not isinstance(ref, dict):
            continue
        platform = ref.get("platform", "")
        source = resolve(ref, platform)
        if source is None or (phrase and phrase not in source[1]):
            continue
        row, text = source
        if row.message_id in seen:
            continue
        source_ids.append(row.message_id)
        examples.append(_example(row, text, is_source=True))
        seen.add(row.message_id)
        # At most the adjacent reviewed turns, clearly distinguished from source evidence.
        for context in ref.get("context_refs", [])[:2]:
            if not isinstance(context, dict):
                continue
            resolved = resolve(context, platform)
            if resolved is not None and resolved[0].message_id not in seen and len(examples) < MAX_EXAMPLES:
                examples.append(_example(*resolved, is_source=False))
                seen.add(resolved[0].message_id)
        if len(examples) >= MAX_EXAMPLES:
            break
    return examples, source_ids


def _record(
    kind: str,
    record_id: int,
    phrase: str,
    meaning: str,
    situation: str,
    examples: List[Dict[str, Any]],
    source_ids: List[str],
) -> Dict[str, Any]:
    item = {
        "id": f"{kind}:{record_id}",
        "kind": kind,
        "phrase": phrase,
        "meaning": meaning,
        "situation": situation,
        "examples": examples,
        "source_ids": source_ids,
    }
    item["version"] = text_digest(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return item


def _enrich_unambiguous_matches(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join complementary fields only when literal matches have one distinct meaning."""
    jargon_by_phrase: Dict[str, List[Dict[str, Any]]] = {}
    for item in records:
        if item["kind"] == "jargon":
            jargon_by_phrase.setdefault(item["phrase"], []).append(item)
    for item in records:
        if item["kind"] != "expression":
            continue
        matches = jargon_by_phrase.get(item["phrase"], [])
        meanings = {match["meaning"] for match in matches}
        if len(meanings) != 1:
            continue
        # A shared spelling is an association, not proof of the same conversational sense.
        item["meaning"] = next(iter(meanings))
        item["meaning_source"] = "linked_jargon_not_situation_verified"
        item["related_ids"] = sorted(match["id"] for match in matches)
        item["related_versions"] = {match["id"]: match["version"] for match in matches}
        # A linked definition does not verify an expression's situation. Keep
        # each record's own examples rather than borrowing another sense's use.
        item["version"] = text_digest(
            json.dumps(
                {key: value for key, value in item.items() if key != "version"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return records


def _load_catalog(session_id: str) -> List[Dict[str, Any]]:
    if not session_id:
        return []
    expression_use, _ = ExpressionConfigUtils.get_expression_config_for_chat(session_id)
    jargon_use, _ = JargonConfigUtils.get_jargon_config_for_chat(session_id)
    result = []
    with get_db_session(auto_commit=False) as db:
        if expression_use:
            candidates = maisaka_expression_selector._load_all_expression_candidates(session_id)
            provenance = ExpressionProvenance(Path(ROOT_PATH) / "data" / "joint_learning.sqlite3").load(session_id)
            rows_by_id = (
                {
                    row.id: row
                    for row in db.exec(
                        select(Expression).where(col(Expression.id).in_([candidate["id"] for candidate in candidates]))
                    ).all()
                }
                if candidates
                else {}
            )
            for candidate in candidates:
                record_id = candidate["id"]
                if get_review_state(record_id)["rejected"]:
                    continue
                row = rows_by_id.get(record_id)
                if row is None or (row.situation, normalize_expression_style_for_learning(row.style)) != (
                    candidate["situation"],
                    candidate["style"],
                ):
                    continue
                refs = provenance.get(record_id, {})
                identity = expression_identity(row.situation, row.style, row.create_time.isoformat())
                examples, source_ids = _resolve_examples(
                    db,
                    session_id,
                    refs.get("sources", []) if refs.get("identity") == identity else [],
                )
                result.append(
                    _record(
                        "expression", record_id, candidate["style"], "", candidate["situation"], examples, source_ids
                    )
                )
        if jargon_use:
            scopes, global_share = JargonConfigUtils.resolve_jargon_group_scope(session_id)
            rows = db.exec(select(Jargon).where(col(Jargon.is_jargon).is_(True))).all()
            for row in rows:
                if row.id is None or not row.content.strip() or not row.meaning.strip():
                    continue
                if not (row.is_global or global_share or _jargon_in_scope(row.session_id_dict, scopes)):
                    continue
                try:
                    groups = json.loads(row.evidence_messages) if row.evidence_messages else []
                except (ValueError, TypeError):
                    groups = []
                refs = (
                    [ref for group in groups if isinstance(group, list) for ref in group]
                    if isinstance(groups, list)
                    else []
                )
                examples, source_ids = _resolve_examples(db, session_id, refs, phrase=row.content)
                result.append(_record("jargon", row.id, row.content, row.meaning, "", examples, source_ids))
    return _enrich_unambiguous_matches(result)


async def load_group_expression_catalog(session_id: str) -> List[Dict[str, Any]]:
    """Return current enabled records, never an unfiltered/stale index snapshot."""
    return await asyncio.to_thread(_load_catalog, session_id)
