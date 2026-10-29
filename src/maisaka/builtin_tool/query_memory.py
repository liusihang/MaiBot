"""query_memory 内置工具。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import asyncio
import json
import re

from src.common.logger import get_logger
from src.config.config import global_config
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.maisaka.tool_result_store import tool_result_store
from src.maisaka.utils.tool_post_execution import with_memory_feedback_task
from src.person_info.person_info import resolve_person_id_for_memory
from src.services.memory_service import MemoryHit, MemorySearchResult, memory_service

from .context import BuiltinToolRuntimeContext

logger = get_logger("maisaka_builtin_query_memory")

_ALLOWED_QUERY_MODES = {"search", "time", "hybrid", "episode", "aggregate"}
_ISO_QUERY_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?$")
REPLYER_MEMORY_REFERENCE_MARKER = "【长期记忆检索结果-内部参考】"
MEMORY_PREVIEW_MAX_CHARS = 2400
MEMORY_REFERENCE_MAX_CHARS = 2200
MEMORY_STRUCTURED_MAX_CHARS = 8000
MEMORY_PREVIEW_MAX_HITS = 5
MEMORY_HIT_PREVIEW_CHARS = 320


def get_tool_spec(*, enabled: bool = True) -> ToolSpec:
    """获取 query_memory 工具声明。"""

    return ToolSpec(
        name="query_memory",
        description=(
            "检索长期记忆。涉及回忆、过往经历时可以检索。无命中不会自动扩大人物或时间范围；"
            "结果是带来源的有限预览，必要时用 read_tool_result 读取原文。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "关键词或问题；非纯时间检索必填。",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回条数。",
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "检索路由：search按语义检索事实、偏好等记忆；time按时间范围检索；"
                        "episode检索经历；aggregate综合检索；hybrid同时使用语义与时间范围。"
                        "time和hybrid至少提供time_start或time_end；没有时间条件时使用search。"
                    ),
                    "enum": sorted(_ALLOWED_QUERY_MODES),
                    "default": "search",
                },
                "person_name": {
                    "type": "string",
                    "description": "人物名；用于定向过滤。",
                },
                "time_start": {
                    "type": "string",
                    "description": "起始时间，仅接受YYYY/MM/DD或YYYY/MM/DD HH:mm。",
                },
                "time_end": {
                    "type": "string",
                    "description": "结束时间，仅接受YYYY/MM/DD或YYYY/MM/DD HH:mm。",
                },
                "respect_filter": {
                    "type": "boolean",
                    "description": "是否遵守记忆过滤规则；默认true，模糊来源或整体印象可false。",
                    "default": True,
                },
            },
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
        enabled=enabled,
    )


def _normalize_optional_time(raw_value: Any) -> str | float | None:
    """归一化可选时间参数。"""

    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        time_text = raw_value.strip()
        if not time_text:
            return None
        if _ISO_QUERY_TIME_RE.fullmatch(time_text):
            return time_text.replace("-", "/", 2)
        return time_text
    if isinstance(raw_value, (float, int)):
        return float(raw_value)

    time_text = str(raw_value).strip()
    if not time_text:
        return None
    return time_text


async def _resolve_person_id(
    *,
    person_name: str,
    platform: str,
    user_id: str,
    group_id: str,
) -> Tuple[str, str]:
    """显式人物称呼按持久映射解析；当前私聊 ID 只用于未指定人物的查询。"""

    clean_person_name = str(person_name or "").strip()
    if clean_person_name:
        result = await memory_service.profile_admin(
            action="resolve_identity",
            identifier=clean_person_name,
        )
        if not result.get("success"):
            raise RuntimeError(f"人物身份解析失败: {result.get('error', '')}")
        return str(result["person_id"]), clean_person_name

    if not group_id and platform and user_id:
        person_id = resolve_person_id_for_memory(
            platform=platform,
            user_id=user_id,
        )
        if person_id:
            return person_id, clean_person_name

    return "", clean_person_name


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value if value is not None else "").strip()
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _preview_text(value: Any, max_chars: int) -> str:
    """结构化字段预算包含 JSON 转义，控制字符不能绕过整包上限。"""
    text = _clip_text(value, max_chars)
    if len(json.dumps(text, ensure_ascii=False)) <= max_chars:
        return text
    return _clip_text(text, max(1, (max_chars - 2) // 6))


def _is_evidence_hit(hit: MemoryHit) -> bool:
    """仅排除明确标记的阈值补齐候选，不拿语义分数裁剪时间/经历检索。"""
    metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
    evidence = metadata.get("retrieval_evidence")
    if isinstance(evidence, dict) and (evidence.get("backfilled") is True or evidence.get("passed") is False):
        return False
    return bool(hit.content.strip())


def _preview_identifier(value: Any, max_chars: int = 160) -> Optional[str]:
    """溯源 ID/URI 必须完整；过长时明确省略，不能造出不存在的前缀 ID。"""
    text = str(value if value is not None else "")
    return text if len(json.dumps(text, ensure_ascii=False)) <= max_chars else None


def _preview_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """预览只带有限正文和溯源字段；完整元数据保留在会话内原始结果中。"""
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    preview_metadata: Dict[str, Any] = {}
    omitted_fields = []
    for key in ("source", "source_type", "chat_id", "event_time", "event_time_start", "event_time_end"):
        if key in metadata:
            value = _preview_identifier(metadata[key])
            if value is None:
                omitted_fields.append(f"metadata.{key}")
            else:
                preview_metadata[key] = value
    for key in ("person_ids", "participants", "source_branches", "evidence_message_ids"):
        values = metadata.get(key)
        if isinstance(values, list):
            identifiers = [_preview_identifier(value, 80) for value in values[:4]]
            preview_metadata[key] = [value for value in identifiers if value is not None]
            if len(values) > 4 or None in identifiers:
                omitted_fields.append(f"metadata.{key}")
    identity = {}
    for key in ("hash", "episode_id", "type", "source"):
        value = _preview_identifier(hit.get(key))
        identity[key] = value or ""
        if value is None:
            omitted_fields.append(key)
    return {
        **identity,
        "title": _preview_text(hit.get("title"), 160),
        "content": _preview_text(hit.get("content"), MEMORY_HIT_PREVIEW_CHARS),
        "score": hit.get("score", 0.0),
        "metadata": preview_metadata,
        "omitted_fields": omitted_fields,
    }


def _bounded_scope(scope: Dict[str, Any]) -> Dict[str, Any]:
    preview = {
        key: value if isinstance(value, (bool, int, float)) or value is None else _preview_identifier(value)
        for key, value in scope.items()
    }
    omitted = [key for key, value in preview.items() if value is None and scope[key] is not None]
    if omitted:
        preview["omitted_fields"] = omitted
    return preview


def _build_result_preview(raw: Dict[str, Any], *, result_id: str, total_chars: int) -> Dict[str, Any]:
    """为历史、落库和 replyer 提供同一份有界回执，不重复携带整包原文。"""
    preview = {
        key: raw[key]
        for key in (
            "success",
            "filtered",
            "status",
            "mode",
            "effective_mode",
            "limit",
            "respect_filter",
            "hit_count",
            "candidate_only_count",
            "primary_hit_count",
            "fallback_applied",
        )
    }
    for key in ("query", "person_name", "error"):
        preview[key] = _preview_text(raw.get(key), 300 if key in {"query", "error"} else 160)
    omitted_fields = []
    for key in ("person_id", "chat_id", "user_id", "group_id"):
        preview[key] = _preview_identifier(raw.get(key))
        if preview[key] is None:
            omitted_fields.append(key)
    for key in ("time_start", "time_end"):
        preview[key] = (
            raw[key] if isinstance(raw[key], (float, int)) or raw[key] is None else _preview_identifier(raw[key], 64)
        )
        if preview[key] is None and raw[key] is not None:
            omitted_fields.append(key)
    preview.update(
        requested_scope=_bounded_scope(raw["requested_scope"]),
        effective_scope=_bounded_scope(raw["effective_scope"]),
        hits=[_preview_hit(hit) for hit in raw["hits"][:MEMORY_PREVIEW_MAX_HITS]],
        result_id=result_id,
        reader_tool="read_tool_result",
        total_chars=total_chars,
        bounded_preview=True,
        truncated=True,
        omitted_fields=omitted_fields,
    )
    # 元数据字段虽逐项限制，仍以整包上限兜住多来源、多人参与的合法大结果。
    while preview["hits"] and len(json.dumps(preview, ensure_ascii=False)) > MEMORY_STRUCTURED_MAX_CHARS - 100:
        preview["hits"].pop()
    preview["preview_hit_count"] = len(preview["hits"])
    preview["truncated"] = bool(raw["excluded_candidates"]) or any(
        raw[key] != value for key, value in preview.items() if key in raw
    )
    return preview


def _build_success_content(structured_content: Dict[str, Any], *, max_chars: int = MEMORY_PREVIEW_MAX_CHARS) -> str:
    """构造带状态、来源和回读入口的有限预览。"""
    status = structured_content["status"]
    if status == "filtered":
        lines = ["当前请求被聊天过滤策略跳过，未执行长期记忆检索。"]
    elif status == "no_match":
        lines = ["未找到满足当前人物、时间和聊天范围的长期记忆；未自动扩大检索范围。"]
    else:
        lines = [f"长期记忆命中 {structured_content['hit_count']} 条（有限预览，仅供核对，不代表事实已证实）。"]
    if structured_content["mode"] != structured_content["effective_mode"]:
        lines.append(f"为保留时间条件，实际模式：{structured_content['effective_mode']}。")
    if structured_content["candidate_only_count"]:
        lines.append(f"另有 {structured_content['candidate_only_count']} 条候选未达到可用证据要求，不作为答案依据。")
    for index, hit in enumerate(structured_content["hits"], start=1):
        source = (
            "来源见原文"
            if "metadata.source" in hit["omitted_fields"]
            else hit["metadata"].get("source") or hit["source"] or "未标注来源"
        )
        source_id = hit["hash"] or hit["episode_id"] or "ID见原文"
        lines.append(f"{index}. [{hit['type']} | {source} | {source_id}] {hit['content']}")
    pointer = f'\n[原始结果 {structured_content["result_id"]}；需要细节时调用 read_tool_result(result_id="{structured_content["result_id"]}")]'
    return _clip_text("\n".join(lines), max_chars - len(pointer)) + pointer


def _build_replyer_memory_reference(structured_content: Dict[str, Any]) -> str:
    """构造自动透传给 replyer 的长期记忆参考。"""
    if structured_content.get("status") != "matched" or not structured_content.get("hits"):
        return ""
    prefix = REPLYER_MEMORY_REFERENCE_MARKER + "\n"
    return prefix + _build_success_content(structured_content, max_chars=MEMORY_REFERENCE_MAX_CHARS - len(prefix))


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """执行 query_memory 内置工具。"""

    runtime = tool_ctx.runtime
    chat_stream = runtime.chat_stream

    clean_query = str(invocation.arguments.get("query") or "").strip()
    mode = str(invocation.arguments.get("mode") or "search").strip().lower() or "search"
    if mode not in _ALLOWED_QUERY_MODES:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"不支持的检索模式：{_clip_text(mode, 80)}。可选值：search/time/hybrid/episode/aggregate。",
        )

    default_limit = max(1, global_config.a_memorix.integration.memory_query_default_limit)
    try:
        limit = int(invocation.arguments.get("limit", default_limit) or default_limit)
    except (TypeError, ValueError):
        limit = default_limit
    limit = max(1, min(limit, 20))

    time_start = _normalize_optional_time(invocation.arguments.get("time_start"))
    time_end = _normalize_optional_time(invocation.arguments.get("time_end"))
    if not clean_query and time_start is None and time_end is None:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "query_memory 需要提供 query，或至少提供 time_start/time_end 中的一个。",
        )

    session_id = str(runtime.session_id or "").strip()
    if not session_id:
        return tool_ctx.build_failure_result(invocation.tool_name, "长期记忆检索缺少真实聊天会话，无法确认检索范围。")
    if any(value and value != session_id for value in (invocation.session_id, context.session_id if context else "")):
        return tool_ctx.build_failure_result(invocation.tool_name, "记忆检索的调用与执行上下文 session_id 不一致。")
    platform = str(chat_stream.platform or "").strip()
    user_id = str(chat_stream.user_id or "").strip()
    group_id = str(chat_stream.group_id or "").strip()
    try:
        person_id, person_name = await _resolve_person_id(
            person_name=str(invocation.arguments.get("person_name") or ""),
            platform=platform,
            user_id=user_id,
            group_id=group_id,
        )
    except RuntimeError as exc:
        return tool_ctx.build_failure_result(invocation.tool_name, _clip_text(exc, 1000))
    if person_name and not person_id:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"无法唯一确认人物 {_clip_text(person_name, 160)}，请使用明确的人物 ID 或不重名的称呼。",
        )
    respect_filter = bool(invocation.arguments.get("respect_filter", True))
    effective_mode = mode
    # search 不应用时间条件，aggregate 的语义分支也不受时间约束。
    # 显式时间条件只能收窄路由，不能默默扩大为不受时间限制的搜索。
    if (time_start is not None or time_end is not None) and mode in {"search", "aggregate"}:
        effective_mode = "hybrid" if clean_query else "time"

    logger.info(
        f"{runtime.log_prefix} 触发长期记忆检索工具: "
        f"mode={mode} query={clean_query!r} person_name={person_name!r} person_id={person_id!r}"
    )
    try:
        result: MemorySearchResult = await memory_service.search(
            clean_query,
            limit=limit,
            mode=effective_mode,
            chat_id=session_id,
            person_id=person_id,
            time_start=time_start,
            time_end=time_end,
            respect_filter=respect_filter,
            user_id=user_id,
            group_id=group_id,
        )
    except Exception as exc:
        logger.exception(f"{runtime.log_prefix} 长期记忆检索执行异常: {exc}")
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"长期记忆检索失败：{_clip_text(exc, 1000)}",
        )
    # min_results 召回补齐仍对自动回忆有效，但显式查证不能把它计为证据命中。
    visible_hits = result.hits if result.success and not result.filtered else []
    evidence_hits = [hit for hit in visible_hits if _is_evidence_hit(hit)]
    candidate_hits = [hit for hit in visible_hits if not _is_evidence_hit(hit)]
    status = (
        "error" if not result.success else "filtered" if result.filtered else "matched" if evidence_hits else "no_match"
    )
    requested_scope = {
        "chat_id": session_id,
        "person_id": person_id,
        "time_start": time_start,
        "time_end": time_end,
        "mode": mode,
        "respect_filter": respect_filter,
    }
    raw_content: Dict[str, Any] = {
        "success": result.success,
        "filtered": result.filtered,
        "error": result.error,
        "status": status,
        "hits": [hit.to_dict() for hit in evidence_hits],
        # 不保存被排除候选的正文，避免分页/关键词回读脱离外层标注后被误用。
        "excluded_candidates": [
            {
                "hash": hit.hash_value,
                "episode_id": hit.episode_id,
                "reason": "relevance_threshold_not_met" if hit.content.strip() else "empty_content",
            }
            for hit in candidate_hits
        ],
        "candidate_note": "excluded_candidates 仅保留被排除候选的 ID 和原因，不是回答依据；正文未保存。",
        "hit_count": len(evidence_hits),
        "candidate_only_count": len(candidate_hits),
        "query": clean_query,
        "mode": mode,
        "effective_mode": effective_mode,
        "limit": limit,
        "chat_id": session_id,
        "person_name": person_name,
        "person_id": person_id,
        "time_start": time_start,
        "time_end": time_end,
        "respect_filter": respect_filter,
        "user_id": user_id,
        "group_id": group_id,
        "fallback_applied": False,
        "primary_hit_count": len(visible_hits),
        "requested_scope": requested_scope,
        "effective_scope": {**requested_scope, "mode": effective_mode},
        "scope_policy": "真实聊天及其已配置共享范围；未移除人物或时间条件。",
    }
    raw_text = await asyncio.to_thread(json.dumps, raw_content, ensure_ascii=False, default=str)
    result_id = tool_result_store.save(session_id, invocation.tool_name, raw_text)
    structured_content = _build_result_preview(raw_content, result_id=result_id, total_chars=len(raw_text))
    metadata: Dict[str, Any] = {
        "bounded_preview": True,
        "result_id": result_id,
        "reader_tool": "read_tool_result",
        "total_chars": len(raw_text),
        "truncated": structured_content["truncated"],
    }

    if not result.success:
        error_message = _clip_text(result.error, 1000) or "长期记忆检索失败。"
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            error_message,
            structured_content=structured_content,
            metadata=metadata,
        )

    content = _build_success_content(structured_content)
    if evidence_hits:
        metadata = with_memory_feedback_task(metadata)
    replyer_memory_reference = _build_replyer_memory_reference(structured_content)
    if replyer_memory_reference:
        metadata["replyer_memory_reference"] = replyer_memory_reference

    return tool_ctx.build_success_result(
        invocation.tool_name,
        content,
        structured_content=structured_content,
        metadata=metadata,
    )
