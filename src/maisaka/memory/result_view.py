"""记忆工具的正文视图；检索诊断保留在原始回执中。"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo
import json
import re


def parse_memory_result(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("hits"), list):
        return None
    return value


def _time_text(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OverflowError, OSError):
            return ""
    return value.strip()[:80] if isinstance(value, str) else ""


def _source_text(hit: dict[str, Any], metadata: dict[str, Any]) -> str:
    labels = []
    for key in ("source_name", "chat_name", "group_name", "person_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in labels:
            labels.append(value.strip()[:160])
    source = str(metadata.get("source") or hit.get("source") or "")
    source_type = str(metadata.get("source_type") or metadata.get("kind") or source.split(":", 1)[0])
    readable = {
        "chat_summary": "群聊摘要", "person_fact": "人物事实", "chat_stream": "聊天记录",
        "episode": "情景记忆", "paragraph_search": "记忆段落", "dual_vector_pool": "记忆段落",
    }.get(source_type, "")
    if readable and readable not in labels:
        labels.append(readable)
    elif source.startswith(("https://", "http://")):
        labels.append(source if len(source) <= 500 else "外部资料")
    elif source and not labels and ":" not in source and source_type not in {
        "fusion_rrf", "paragraph_vector_pool", "paragraph_sparse", "graph_vector_pool",
    }:
        labels.append(source[:160])
    return " · ".join(labels)


def _entries(raw: dict[str, Any]) -> list[tuple[str, str]]:
    entries = []
    people = {item["person_id"]: item for item in raw.get("people", [])}
    for hit in raw.get("hits", []):
        if not isinstance(hit, dict):
            continue
        body = str(hit.get("content") or "").strip()
        title = str(hit.get("title") or "").strip()
        if not body and not title:
            continue
        metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        kind = {"paragraph": "记忆", "episode": "情景", "relation": "关联事实", "fact": "事实"}.get(
            hit.get("type"), "记忆"
        )
        lines = [f"{len(entries) + 1}. {kind}" + (f"：{title}" if title else "")]
        ids = list(metadata.get("person_ids", [])) if isinstance(metadata.get("person_ids"), list) else []
        if metadata.get("person_id"):
            ids.append(metadata["person_id"])
        names = list(dict.fromkeys(
            people[pid]["primary_name"] or people[pid]["qq_name"] or people[pid]["user_id"]
            for pid in ids if pid in people
        ))
        if names:
            lines.append("人物：" + "、".join(names))
        source = _source_text(hit, metadata)
        if source:
            lines.append("来源：" + source)
        start = _time_text(metadata.get("event_time_start") or metadata.get("event_time"))
        end = _time_text(metadata.get("event_time_end"))
        if start or end:
            lines.append("时间：" + (f"{start} 至 {end}" if start and end and start != end else start or end))
        if body:
            lines.append("内容：" + body)
        entries.append(("\n".join(lines), (title + "\n" + body).casefold()))
    return entries


def memory_result_preview(raw: dict[str, Any], result_id: str, *, max_chars: int = 2400) -> str:
    """初次返回正文与可继续读取的定位，后台回执不参与展示。"""
    if raw.get("status") == "filtered":
        return "当前聊天范围未启用这类记忆检索。"
    if raw.get("status") == "error":
        return "记忆检索失败：" + str(raw.get("error") or "读取失败")[:300]
    entries = _entries(raw)
    if not entries:
        return "当前条件下未找到相关记忆。"
    pointer = f'\n\n续读：read_tool_result(result_id="{result_id}")'
    available = max(0, max_chars - len(pointer))
    blocks = []
    used = 0
    for block, _ in entries:
        separator = 2 if blocks else 0
        if used + separator + len(block) > available:
            if not blocks:
                blocks.append(block[:max(0, available - 10)] + "\n（正文可续读）")
            break
        blocks.append(block)
        used += separator + len(block)
    return ("\n\n".join(blocks) + pointer)[:max_chars]


def memory_result_page(raw: dict[str, Any], offset: int, limit: int, keyword: str = "") -> dict[str, Any]:
    """在标题和正文中定位；普通条目完整返回，超长单条按可读正文坐标续读。"""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 2800:
        raise ValueError("Invalid evidence pagination")
    if len(keyword) > 256:
        raise ValueError("Evidence keyword is too long")
    records, position = [], 0
    for block, searchable in _entries(raw):
        text = block + "\n\n"
        records.append((position, position + len(text), text, searchable))
        position += len(text)
    selected = [r for r in records if r[1] > offset and (not keyword or keyword.casefold() in r[3])]
    if not selected:
        return {"text": "", "offset": min(offset, position), "next_offset": None,
                "total_chars": position, "matched": False}
    parts, remaining = [], limit
    first_offset, end_offset = selected[0][0], selected[0][0]
    for start, end, block, _ in selected:
        read_start = max(offset, start)
        if keyword and not parts and len(block) > limit and read_start == start:
            # 超长条目的首次关键词读取落在关键词附近；普通大小的条目整条返回。
            found = re.search(re.escape(keyword), block, re.IGNORECASE)
            if found is not None:
                read_start = max(start, start + found.start() - min(250, limit // 4))
        chunk = block[read_start - start:]
        if parts and len(chunk) > remaining:
            break
        if not parts:
            first_offset = read_start
        parts.append(chunk[:remaining])
        taken = min(len(chunk), remaining)
        remaining -= taken
        end_offset = read_start + taken
        if end_offset < end or not remaining:
            break
    more = any(end > end_offset for _, end, _, _ in selected)
    return {"text": "".join(parts), "offset": first_offset,
            "next_offset": end_offset if more else None, "total_chars": position, "matched": True}


def format_evidence_page(page: dict[str, Any], result_id: str) -> str:
    """模型只接收正文、空结果说明与必要的下一页坐标。"""
    text = str(page.get("text") or "")
    if not text:
        return "正文或标题中没有这个关键词，或已读取完毕。"
    next_offset = page.get("next_offset")
    if next_offset is not None:
        return text.rstrip() + f'\n\n续读：read_tool_result(result_id="{result_id}", offset={next_offset})'
    return text.rstrip()
