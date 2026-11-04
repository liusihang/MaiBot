"""Save a sourced group note or a person's group aliases, with a storage receipt."""

from __future__ import annotations

import asyncio

from src.config.config import global_config
from src.core.tooling import ToolSpec
from src.common.message_repository import find_messages
from src.common.person_names import person_name_records
from src.person_info.person_info import get_person_id
from src.chat.utils.utils import is_bot_self
from src.services.memory_service import memory_service
from src.services.memory_records import message_identity, user_text, memory_external_id


def get_tool_spec():
    return ToolSpec(
        name="remember_memory",
        description=(
            "保存你判断值得长期保留的称呼、人物信息或群聊语境。查询已有记忆使用 query_memory。"
            "正文直接记录具体发言、对象和后续互动；后续如何回应由你结合当前语境判断。"
            "evidence_message_ids 填写当前群支持该内容的真实用户消息ID。"
            "subject_name 优先填人物画像中的主称呼，其次填QQ昵称；aliases 填原话明确给出的该人别名。"
            "群友称呼、第三方评价和本人偏好按各自发言归属保存。"
            "返回 saved、stored_ids 和 alias_saved；成功回执后确认已保存，失败按具体原因处理。"
        ),
        parameters_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2000,
                    "description": "要保存的简洁内容；群内约定注明语境，保留人物指代和条件",
                },
                "evidence_message_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                    "description": "支持该内容的真实用户消息ID；可合并前后几条确定指代",
                },
                "subject_name": {
                    "type": "string",
                    "default": "",
                    "description": "人物信息的对象，优先主称呼，其次QQ昵称；群内名单/一般约定留空",
                },
                "aliases": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 40},
                    "maxItems": 12,
                    "default": [],
                    "description": "原话明确给出的对象别名；不填推测称呼，不把其他账号合并为同一身份",
                },
            },
            "required": ["content", "evidence_message_ids"],
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
        enabled=bool(global_config.a_memorix.integration.enable_memory_query_tool),
    )


async def handle_tool(tool_ctx, invocation, context=None):
    args = invocation.arguments
    content = str(args.get("content", "") or "").strip()
    ids = args.get("evidence_message_ids")
    subject = str(args.get("subject_name", "") or "").strip()
    aliases = args.get("aliases") or []

    def fail(message: str, payload=None):
        return tool_ctx.build_failure_result("remember_memory", message, structured_content=payload)

    def reject(message: str):
        return tool_ctx.build_failure_result(
            "remember_memory", message, metadata={"execution_outcome": "not_executed"},
        )

    if not content or len(content) > 2000 or not isinstance(ids, list) or not 1 <= len(ids) <= 8:
        return reject("需要1–8个真实用户消息ID和不超过2000字的 content。")
    if not isinstance(aliases, list) or any(not isinstance(a, str) or not 1 <= len(a.strip()) <= 40 for a in aliases):
        return reject("aliases 必须是原话中的简短别名列表。")
    rt = tool_ctx.runtime
    sid = rt.session_id
    stream = rt.chat_stream
    evidence = []
    for mid in ids:
        rows = await asyncio.to_thread(find_messages, session_id=sid, message_id=str(mid), limit=1)
        m = rows[0] if rows else rt.find_source_message_by_id(str(mid))
        if m is None or message_identity(m)[0] != sid:
            return reject(f"消息 {mid} 不在当前聊天中，请使用真实消息ID。")
        if is_bot_self(m.platform, message_identity(m)[2]) or not user_text(m):
            return reject(f"消息 {mid} 是机器人或没有用户原文，不能作为记忆依据。")
        evidence.append(m)
    body = "\n".join(user_text(m) for m in evidence)
    person_id = ""
    if aliases and not subject:
        return reject("保存人物别名时请提供该人的真实 subject_name。")
    if subject:
        candidates = await asyncio.to_thread(find_messages, session_id=sid, limit=5000)
        names = await asyncio.to_thread(
            person_name_records, platform=evidence[0].platform,
            user_ids={str(m.message_info.user_info.user_id) for m in candidates},
        )
        primary_names = {record["user_id"]: record["primary_name"] for record in names.values()}
        matches = {}
        for m in candidates:
            info = m.message_info.user_info
            if subject in {
                primary_names.get(str(info.user_id), ""), str(info.user_nickname or ""), str(info.user_cardname or ""),
            } and not is_bot_self(
                m.platform, info.user_id
            ):
                matches[str(info.user_id)] = m
        if len(matches) != 1:
            return reject("subject_name 未唯一匹配当前群成员；请用其主称呼或QQ昵称，或先把内容作为群内约定保存。")
        uid, target = next(iter(matches.items()))
        if not any(message_identity(m)[2] == uid for m in evidence) and subject not in body:
            return reject("证据原话没有明确指出该对象；补充含对象名字的消息，或先保存为群内约定。")
        person_id = get_person_id(target.platform, uid)
        if any(a.strip() not in body for a in aliases):
            return reject("某个别名未出现在用户原话中，请删除推测别名。")
    actual_ids = [str(m.message_id) for m in evidence]
    evidence_rows = [
        {
            "message_id": str(m.message_id),
            "user_id": message_identity(m)[2],
            "speaker": m.message_info.user_info.user_cardname or m.message_info.user_info.user_nickname,
            "text": user_text(m),
        }
        for m in evidence
    ]
    text = (
        content
        + "\n\n来源：\n"
        + "\n".join(f"{row['speaker']}：{row['text']}" for row in evidence_rows)
    )
    result = await memory_service.ingest_text(
        external_id=memory_external_id(sid, content, actual_ids),
        source_type="chat_summary",
        text=text,
        chat_id=sid,
        person_ids=[person_id] if person_id else [],
        participants=[subject] if subject else [],
        tags=["explicit_memory", "group_context"],
        metadata={
            "evidence_message_ids": actual_ids,
            "evidence": evidence_rows,
            "evidence_source": "explicit_user_request",
            "writeback_source": "remember_memory",
            "subject_name": subject,
            "aliases": aliases,
        },
        respect_filter=True,
        group_id=str(stream.group_id or ""),
        user_id=str(stream.user_id or ""),
    )
    saved = result.success and result.detail != "chat_filtered" and bool(result.stored_ids or result.skipped_ids)
    receipt = {
        "saved": saved,
        "stored_ids": result.stored_ids or result.skipped_ids,
        "evidence_message_ids": actual_ids,
        "subject_name": subject,
        "alias_saved": not bool(aliases),
    }
    if not saved:
        return fail("内容未保存：" + result.detail + "。不能确认已记住。", receipt)
    if aliases:
        previous = await memory_service.profile_admin(action="get_aliases", person_id=person_id)
        if not previous.get("success"):
            return fail("记忆正文已保存，别名读取失败；暂不能确认别名已生效。", receipt)
        merged = list(dict.fromkeys([*previous.get("effective_aliases", []), *[a.strip() for a in aliases]]))
        update = await memory_service.profile_admin(
            action="set_aliases",
            person_id=person_id,
            aliases=merged,
            updated_by="group_evidence:" + ",".join(actual_ids),
            source="explicit_user_request",
        )
        receipt["alias_saved"] = bool(update.get("success"))
        if not receipt["alias_saved"]:
            return fail("正文已保存，别名更新失败：" + str(update.get("error", "")), receipt)
    from src.maisaka.memory.heuristic_injector import heuristic_memory_injector

    heuristic_memory_injector.clear_session_reference(sid)
    return tool_ctx.build_success_result(
        "remember_memory", "已保存，可在超出上下文后通过 query_memory 查询。", structured_content=receipt
    )
