"""Evidence and identity helpers shared by automatic and explicit memory writes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import hashlib
import re
import unicodedata

from src.common.data_models.message_component_data_model import TextComponent


def message_identity(message):
    info = getattr(message, "message_info", None)
    user = getattr(info, "user_info", None)
    group = getattr(info, "group_info", None)
    session = getattr(message, "session", None)
    return (
        str(getattr(message, "session_id", "") or getattr(session, "session_id", "") or ""),
        str(
            getattr(group, "group_id", "") or getattr(session, "group_id", "") or getattr(message, "group_id", "") or ""
        ),
        str(getattr(user, "user_id", "") or getattr(session, "user_id", "") or getattr(message, "user_id", "") or ""),
    )


def user_text(message):
    """Read only the author's TextComponents, excluding quoted and forwarded text."""
    raw = getattr(message, "raw_message", None)
    if raw is not None:
        return "\n".join(c.text for c in raw.components if isinstance(c, TextComponent)).strip()
    return str(getattr(message, "processed_plain_text", "") or "").strip()


def normalized(value):
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", str(value)).casefold())


def fact_key(value, person_name=""):
    text = normalized(value)
    for prefix in sorted(
        [normalized(person_name), "目标人物", "该用户", "目标用户", "用户", "本人", "他", "她", "其"],
        key=len,
        reverse=True,
    ):
        if prefix and text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.removeprefix("的").replace("自己", "")


def direct_self_evidence(item, messages, person_name=""):
    """Accept a sourced, declarative self statement; semantic extraction is separate."""
    if not isinstance(item, dict):
        return None
    mid = str(item.get("evidence_message_id", "") or "")
    quote = str(item.get("evidence_quote", "") or "").strip()
    source = next((m for m in messages if str(m.message_id) == mid), None)
    if source is None or not quote:
        return None
    body = user_text(source)
    if normalized(quote) not in normalized(body):
        return None
    # A question to the bot and another person's quoted answer are not self reports.
    sentences = re.split(r"[。！!；;\n]", body)
    sentence = next((s.strip() for s in sentences if normalized(quote) in normalized(s)), "")
    personal_request = bool(re.search(r"(?:别|不要|不想|不希望).{0,20}我", sentence))
    if not sentence or (("?" in sentence or "？" in sentence) and not personal_request):
        return None
    sentence = re.sub(r"^\s*@\S+\s*", "", sentence)
    if not personal_request and not re.match(r"^(?:我(?!们|朋友|同事|爸|妈|家人|的朋友|的同事)|本人|自己)", sentence):
        return None
    if any(s in sentence for s in ["我问你", "我问的是", "我说你", "我让你", "我想问", "我朋友", "我的朋友"]):
        return None
    return source


def memory_external_id(session_id, content, message_ids):
    key = "|".join([session_id, normalized(content), *sorted(set(message_ids))])
    return "explicit_memory:" + hashlib.sha256(key.encode()).hexdigest()


def source_evidence(message: Any) -> Dict[str, Any]:
    """Keep attribution and time separately from quoted context; quotes are never self evidence."""
    record: Dict[str, Any] = {"message_id": str(message.message_id), "text": user_text(message)}
    timestamp = getattr(message, "timestamp", None) or getattr(message, "received_at", None)
    if isinstance(timestamp, datetime):
        record["time"] = timestamp.isoformat()
    elif isinstance(timestamp, (int, float)):
        record["time"] = timestamp
    replies = []
    for component in getattr(getattr(message, "raw_message", None), "components", []):
        if getattr(component, "format_name", "") == "reply":
            replies.append(
                {
                    "message_id": str(component.target_message_id),
                    "sender_id": str(component.target_message_sender_id),
                    "text": str(component.target_message_content or "")[:2000],
                }
            )
    if replies:
        record["reply_to"] = replies
    return record


def fact_classification(item: Dict[str, Any]) -> Dict[str, str] | None:
    """Reuse the existing claim schema and keep temporary evidence out of stable profiles."""
    stability = item.get("stability", "stable")  # Durable pre-upgrade extraction receipts.
    if stability not in {"stable", "temporal", "uncertain"}:
        return None
    section = item.get("profile_section", "stable_facts")
    if stability == "temporal":
        section = "recent_interactions"
    elif stability == "uncertain":
        section = "uncertain_notes"
    elif section not in {"stable_facts", "identity_settings", "interaction_preferences", "relationship_settings"}:
        return None
    return {"authority": "direct_user", "stability": stability, "profile_section": section}


def explicit_correction(text: str) -> bool:
    """Require an explicit change/correction cue before an automatic supersession."""
    return bool(
        re.search(
            r"更正|纠正|说错|不再|改成|改为|改了|搬到|搬去|转到|换成|不是|之前.{0,40}(?:现在|其实)|不是.{0,40}(?:而是|是)|no longer|correction|used to|actually|moved to|switched to|訂正|今は|ではなく",
            text,
            re.IGNORECASE,
        )
    )
