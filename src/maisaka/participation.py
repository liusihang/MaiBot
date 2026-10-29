"""Luna decides participation only; the main Agent retains context and authorship."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import time
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt_template
from src.config.config import global_config
from src.llm_models.payload_content.context_item import ContextItemBuilder, RoleType
from src.llm_models.payload_content.resp_format import RespFormat, RespFormatType
from src.services.llm_service import LLMServiceClient

if TYPE_CHECKING:
    from src.maisaka.runtime import MaisakaHeartFlowChatting

logger = get_logger("maisaka_participation")
PARTICIPATION_MODEL = "group-helper"
PARTICIPATION_TIMEOUT_SECONDS = 30
PARTICIPATION_INTERVAL_SECONDS = 30.0
RECENT_MESSAGES = 24
TEXT_LIMIT = 1600
DECISION_SCHEMA = {
    "name": "group_participation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"decision": {"type": "string", "enum": ["engage", "ignore", "uncertain"]}},
        "required": ["decision"],
        "additionalProperties": False,
    },
}


def shorten(text: str) -> str:
    if len(text) <= TEXT_LIMIT:
        return text
    return text[:1000] + "\n[中间较长内容省略]\n" + text[-600:]


def message_record(message, *, is_bot: bool) -> dict:
    user = message.message_info.user_info
    record = {
        "msg_id": str(message.message_id),
        "sender_id": str(user.user_id),
        "sender": user.user_cardname or user.user_nickname or str(user.user_id),
        "is_bot": is_bot,
        "text": shorten(message.processed_plain_text or ""),
    }
    when = getattr(message, "received_at", None)
    if when is not None:
        record["time"] = when.isoformat()
    replies = []
    for component in getattr(getattr(message, "raw_message", None), "components", []):
        if getattr(component, "format_name", "") == "reply":
            replies.append(
                {
                    "msg_id": component.target_message_id,
                    "sender_id": component.target_message_sender_id,
                    "sender": component.target_message_sender_cardname or component.target_message_sender_nickname,
                    "text": shorten(component.target_message_content or ""),
                }
            )
    if replies:
        record["reply_to"] = replies
    return record


def build_payload(runtime, messages) -> dict:
    pending_ids = {str(message.message_id) for message in messages}
    recent = []
    for entry in runtime._chat_history:
        original = getattr(entry, "original_message", None)
        if original is None or not entry.count_in_context or str(original.message_id) in pending_ids:
            continue
        record = message_record(original, is_bot=entry.source == "guided_reply")
        # Native history can include refreshed media descriptions absent from the raw plain text.
        record["text"] = shorten(entry.processed_plain_text or record["text"])
        recent.append(record)
    return {
        "bot_names": list(dict.fromkeys([global_config.bot.nickname, *global_config.bot.alias_names])),
        "recent_messages": recent[-RECENT_MESSAGES:],
        "new_messages": [message_record(m, is_bot=False) for m in messages],
    }


def build_context(payload: dict, now: datetime | None = None):
    prompt = load_prompt_template("participation_judge", locale="zh-CN").template
    current = now if now is not None else datetime.now(ZoneInfo("Asia/Shanghai"))
    current = current.astimezone(ZoneInfo("Asia/Shanghai"))
    prompt += f"\n\n当前时间：{current:%Y-%m-%d %H:%M:%S}，星期{'一二三四五六日'[current.weekday()]}。时区 Asia/Shanghai（北京时间，UTC+08:00）。\n"
    return [
        ContextItemBuilder().set_role(RoleType.System).add_text_content(prompt).build(),
        ContextItemBuilder().add_text_content(json.dumps(payload, ensure_ascii=False)).build(),
    ]


def parse_decision(text: str) -> str:
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return "uncertain"
    if not isinstance(obj, dict) or set(obj) != {"decision"}:
        return "uncertain"
    value = obj.get("decision")
    return value if isinstance(value, str) and value in {"engage", "ignore", "uncertain"} else "uncertain"


def quotes_bot(runtime, messages) -> bool:
    bot_id = str(getattr(runtime.chat_stream, "account_id", "") or "")
    own_ids = {
        str(entry.message_id)
        for entry in runtime._chat_history
        if getattr(entry, "source", "") == "guided_reply" and getattr(entry, "message_id", None)
    }
    for message in messages:
        for component in getattr(getattr(message, "raw_message", None), "components", []):
            if getattr(component, "format_name", "") != "reply":
                continue
            if component.target_message_id in own_ids or (bot_id and component.target_message_sender_id == bot_id):
                return True
    return False


class ParticipationJudge:
    def __init__(self, runtime: MaisakaHeartFlowChatting):
        self.runtime = runtime
        self._client = None
        self._last_judgment_started_at: float | None = None

    def cooldown_remaining(self) -> float:
        if self._last_judgment_started_at is None:
            return 0.0
        return max(0.0, PARTICIPATION_INTERVAL_SECONDS - (time.monotonic() - self._last_judgment_started_at))

    def defer_pending_turn(self) -> bool:
        """Leave ordinary messages in the native cache until the next judgment slot."""
        runtime = self.runtime
        remaining = self.cooldown_remaining()
        if remaining <= 0 or not runtime.chat_stream.is_group_session:
            return False
        messages = runtime.message_cache[runtime._last_processed_index :]
        if (
            not messages
            or runtime._has_forced_turn_trigger()
            or runtime._is_focus_mode_active_for_current_chat()
            or runtime._agent_state == runtime._STATE_WAIT
            or any(m.is_at or m.is_mentioned for m in messages)
            or quotes_bot(runtime, messages)
        ):
            return False
        runtime._mark_message_turn_unscheduled()
        runtime._defer_message_turn_check(remaining)
        logger.info(f"{runtime.log_prefix} 参与判断冷却 剩余={remaining:.2f}s 待处理={len(messages)}，到期合并判断")
        return True

    async def should_enter(self, turn) -> bool:
        runtime = self.runtime
        messages = turn.cached_messages
        if (
            not runtime.chat_stream.is_group_session
            or not messages
            or turn.timeout_triggered
            or turn.proactive_triggered
            or turn.logical_turn_id
            or runtime._has_forced_turn_trigger()
            or runtime._is_focus_mode_active_for_current_chat()
            or any(m.is_at or m.is_mentioned for m in messages)
            or quotes_bot(runtime, messages)
        ):
            logger.info(f"{runtime.log_prefix} 参与判断 直接放行：明确指向机器人或已在执行的任务")
            return True
        started = time.monotonic()
        self._last_judgment_started_at = started
        runtime._update_stage_status("参与判断", "Luna 判断连续对话、开放提问与可参与话题")
        try:
            payload = build_payload(runtime, messages)
            if self._client is None:
                self._client = LLMServiceClient(
                    task_name="fast_model", request_type="participation_judge", session_id=runtime.session_id
                )
            context = build_context(payload)
            result = await asyncio.wait_for(
                self._client.generate_response_with_context(
                    context_factory=lambda _: context,
                    options=LLMGenerationOptions(
                        model_name=PARTICIPATION_MODEL,
                        retain_replay=False,
                        response_format=RespFormat(RespFormatType.JSON_SCHEMA, DECISION_SCHEMA),
                    ),
                ),
                timeout=PARTICIPATION_TIMEOUT_SECONDS,
            )
            decision = parse_decision(result.response)
            logger.info(
                f"{runtime.log_prefix} 参与判断 模型={result.model_name} decision={decision} "
                f"新消息={len(messages)} 耗时={time.monotonic() - started:.2f}s"
            )
            return decision != "ignore"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"{runtime.log_prefix} 参与判断失败，交给主 Agent：{type(exc).__name__}")
            return True
