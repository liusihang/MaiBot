"""Keyset import of persisted group messages, not the main agent's sampled tail."""

from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple

import json


def source_fingerprint(row: Any) -> str:
    # Display names, derived captions and temporary outbound message IDs can be
    # updated later. Anchor stable source fields, not delivery annotations;
    # a reused SQLite id must fail shut.
    values = [
        row.id,
        row.session_id,
        row.platform,
        row.user_id,
        row.group_id,
        row.timestamp.isoformat(),
        bytes(row.raw_content).hex(),
    ]
    return sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def read_source_page(
    cursor: int,
    boundary: float,
    anchor: Optional[Dict[str, Any]],
    *,
    limit: int = 500,
) -> Tuple[List[Dict[str, Any]], int, Optional[Dict[str, Any]], bool]:
    from sqlalchemy import func
    from sqlmodel import select
    from src.chat.utils.utils import is_bot_self
    from src.common.database.database import get_db_session
    from src.common.database.database_model import Messages

    with get_db_session(auto_commit=False) as db:
        maximum = db.exec(select(func.max(Messages.id))).one() or 0
        if maximum < cursor:
            raise RuntimeError("Daily topic source database rolled back/reset; reconcile journal before import")
        if cursor:
            last = db.get(Messages, cursor)
            if last is None or anchor is None or source_fingerprint(last) != anchor["fingerprint"]:
                raise RuntimeError("Daily topic source cursor was removed/reused; reconcile journal before import")
        rows = db.exec(select(Messages).where(Messages.id > cursor).order_by(Messages.id).limit(limit)).all()
        records = []
        for row in rows:
            # Database timestamps intentionally use the host's local naive
            # datetime convention, like message_repository; never assume UTC.
            occurred_at = row.timestamp.timestamp()
            if row.group_id and row.session_id and occurred_at >= boundary:
                records.append(
                    {
                        "source_id": "message:" + str(row.id),
                        "message_id": row.message_id,
                        "session_id": row.session_id,
                        "occurred_at": occurred_at,
                        "author": row.user_id,
                        "author_name": row.user_cardname or row.user_nickname,
                        "platform": row.platform,
                        "is_bot": is_bot_self(row.platform, row.user_id),
                        "text": row.processed_plain_text or "[No stored text; media content not inferred]",
                    }
                )
        if rows:
            cursor = rows[-1].id
            anchor = {"row_id": cursor, "fingerprint": source_fingerprint(rows[-1])}
        return records, cursor, anchor, cursor >= maximum
