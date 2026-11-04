"""ID-bound names shared by model input, profiles and the dashboard.

PersonInfo.person_name is the maintained primary name. QQ nicknames are a
display fallback; group cards never decide identity or the displayed name.
Synchronous callers run these database operations in their worker thread.
"""

from copy import copy
from typing import Any, Dict, Iterable

import re

from sqlmodel import col, select

from src.common.database.database import get_db_session
from src.common.database.database_model import PersonInfo


def person_name_records(*, person_ids: Iterable[str] = (), platform: str = "", user_ids: Iterable[str] = ()) -> Dict[str, Dict[str, str]]:
    ids = list(dict.fromkeys(str(value) for value in person_ids if value))
    users = list(dict.fromkeys(str(value) for value in user_ids if value))
    if not ids and not users:
        return {}
    statement = select(PersonInfo.person_id, PersonInfo.platform, PersonInfo.user_id,
                       PersonInfo.person_name, PersonInfo.user_nickname)
    if ids:
        statement = statement.where(col(PersonInfo.person_id).in_(ids))
    else:
        statement = statement.where(PersonInfo.platform == platform, col(PersonInfo.user_id).in_(users))
    with get_db_session(auto_commit=False) as db:
        return {row.person_id: {
            "person_id": row.person_id, "platform": row.platform, "user_id": row.user_id,
            "primary_name": (row.person_name or "").strip(), "qq_name": row.user_nickname.strip(),
        } for row in db.exec(statement).all()}


def display_name(platform: str, user_id: str, qq_name: str = "") -> str:
    from src.chat.utils.utils import is_bot_self
    from src.config.config import global_config

    qq_name = qq_name.strip() if isinstance(qq_name, str) else ""
    if is_bot_self(platform, user_id):
        return global_config.bot.nickname
    records = person_name_records(platform=platform, user_ids=[user_id])
    for record in records.values():
        return record["primary_name"] or qq_name.strip() or record["qq_name"] or user_id
    return qq_name.strip() or user_id


def project_message_names(message: Any) -> Any:
    """Copy only presentation fields; source objects, IDs and media stay intact."""
    from src.common.data_models.message_component_data_model import AtComponent, ForwardNodeComponent, ReplyComponent

    projected = copy(message)
    projected.message_info = copy(message.message_info)
    projected.message_info.user_info = copy(message.message_info.user_info)
    user = projected.message_info.user_info
    user.user_nickname = display_name(message.platform, user.user_id, user.user_nickname or "")
    user.user_cardname = ""

    def components(items):
        result = []
        for item in items:
            if isinstance(item, AtComponent):
                item = copy(item)
                item.target_user_nickname = display_name(message.platform, item.target_user_id, item.target_user_nickname or "")
                item.target_user_cardname = ""
            elif isinstance(item, ReplyComponent):
                item = copy(item)
                item.target_message_sender_nickname = display_name(
                    message.platform, item.target_message_sender_id or "", item.target_message_sender_nickname or "",
                )
                item.target_message_sender_cardname = ""
            elif isinstance(item, ForwardNodeComponent):
                item = copy(item)
                item.forward_components = [copy(node) for node in item.forward_components]
                for node in item.forward_components:
                    node.user_nickname = display_name(message.platform, node.user_id or "", node.user_nickname or "")
                    node.user_cardname = ""
                    node.content = components(node.content)
            result.append(item)
        return result

    projected.raw_message = copy(message.raw_message)
    projected.raw_message.components = components(message.raw_message.components)
    return projected


def profile_display_names(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Render cached profile headers with the current maintained primary name."""
    result = dict(payload)
    items = [dict(item) for item in result.get("items", []) if isinstance(item, dict)]
    profile = dict(result["profile"]) if isinstance(result.get("profile"), dict) else None
    entries = [result, *items, *([profile] if profile is not None else [])]
    records = person_name_records(person_ids=[item.get("person_id", "") for item in entries])
    for item in entries:
        record = records.get(str(item.get("person_id", "")))
        if not record:
            continue
        name = record["primary_name"] or record["qq_name"] or record["user_id"]
        item["person_name"] = name
        item["primary_name"] = name
        for key in ("profile_text", "auto_profile_text", "summary"):
            if isinstance(item.get(key), str):
                item[key] = re.sub(r"(?m)^主称呼[:：].*$", lambda _, current=name: "主称呼: " + current, item[key])
    if "items" in result:
        result["items"] = items
    if profile is not None:
        result["profile"] = profile
    return result


def monitor_display_names(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    platform, user_id = str(result.get("platform", "")), str(result.get("user_id", ""))
    if platform and user_id and "speaker_name" in result:
        result["speaker_name"] = display_name(platform, user_id, str(result.get("qq_name", "")))
    quote = result.get("reply_to")
    if isinstance(quote, dict) and quote.get("user_id"):
        quote = dict(quote)
        quote["sender_name"] = display_name(platform, str(quote["user_id"]), str(quote.get("qq_name", "")))
        result["reply_to"] = quote
    return result


def memory_people(payload: Dict[str, Any]) -> list[Dict[str, str]]:
    ids = []
    for hit in payload.get("hits", []):
        metadata = hit.get("metadata", {})
        person_ids = metadata.get("person_ids", [])
        if isinstance(person_ids, list):
            ids.extend(person_ids)
        if metadata.get("person_id"):
            ids.append(metadata["person_id"])
    return list(person_name_records(person_ids=ids).values())
