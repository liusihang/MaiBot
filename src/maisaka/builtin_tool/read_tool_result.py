"""read_tool_result 内置工具：取回被整理过的工具原始结果。"""

from typing import Any, List, Optional
import re

from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.maisaka.tool_result_store import StoredToolResult, tool_result_store

from .context import BuiltinToolRuntimeContext

READ_TOOL_RESULT_TOOL_NAME = "read_tool_result"
# 一次读取的默认长度与上限，保持在超长结果整理的门槛之内，避免读回来的内容又是一大段
READ_DEFAULT_MAX_CHARS = 2000
READ_MAX_CHARS_LIMIT = 2800
# 按关键词查找时，每处命中前后各带多少字的上下文，以及最多返回几处
KEYWORD_CONTEXT_CHARS = 250
KEYWORD_MAX_MATCHES = 5
READ_MAX_KEYWORD_CHARS = 256
# 原文预算之外，标签和定位信息仍有一个固定的总长度上限。
READ_OUTPUT_MAX_CHARS = READ_MAX_CHARS_LIMIT + 1024


def get_tool_spec() -> ToolSpec:
    """获取 read_tool_result 工具声明。"""

    return ToolSpec(
        name=READ_TOOL_RESULT_TOOL_NAME,
        description=(
            "取回某次工具调用保存的原始结果。记忆/联网工具预览或超长结果的整理结论会附有编号（如 r3）；"
            "当你需要其中没有的细节时，用这个编号分段读取原文，或用 keyword 直接查找相关片段。"
            "已有内容够用时不需要调用。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "工具返回的原始结果编号，例如 r3。",
                    "maxLength": 64,
                    "pattern": r"^r[0-9a-f]+$",
                },
                "keyword": {
                    "type": "string",
                    "description": "可选。只返回原文中包含该关键词的片段及其前后文；用 offset 继续查找，不填则顺序读取。",
                    "maxLength": READ_MAX_KEYWORD_CHARS,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "可选。从原文第几个字开始读，默认 0；上一次读取会告诉你下一段的 offset。",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": READ_MAX_CHARS_LIMIT,
                    "description": f"可选。本次最多读取多少字，默认 {READ_DEFAULT_MAX_CHARS}。",
                },
            },
            "required": ["result_id"],
            "additionalProperties": False,
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_by_offset(stored_result: StoredToolResult, offset: int, max_chars: int) -> str:
    """按偏移顺序读取一段原文。"""

    total_chars = len(stored_result.content)
    start = min(max(offset, 0), total_chars)
    end = min(start + max_chars, total_chars)
    header = f"[{stored_result.result_id} · {stored_result.tool_name[:80]} 原始结果 · 共 {total_chars} 字 · 本段 {start}–{end}]"
    if start >= total_chars:
        return f"{header}\n（offset 已超出原文末尾，没有更多内容。）"
    remaining_chars = total_chars - end
    footer = (
        f"\n（后面还有 {remaining_chars} 字，用 offset={end} 继续读取。）"
        if remaining_chars > 0
        else "\n（已到原文末尾。）"
    )
    return f"{header}\n{stored_result.content[start:end]}{footer}"


def _read_by_keyword(stored_result: StoredToolResult, keyword: str, offset: int, max_chars: int) -> str:
    """按原文坐标检索，关键词、片段数与所有片段共享的字符预算均有限。"""

    content = stored_result.content
    start_offset = min(max(offset, 0), len(content))
    snippets: List[str] = []
    total_matches = 0
    last_snippet_end = start_offset
    remaining = max_chars
    # 在原文上匹配，避免 lower/casefold 改变 Unicode 长度后错移坐标。
    for match in re.finditer(re.escape(keyword), content, re.IGNORECASE):
        if match.start() < start_offset:
            continue
        total_matches += 1
        if match.start() < last_snippet_end or len(snippets) >= KEYWORD_MAX_MATCHES or remaining <= 0:
            continue
        before = min(KEYWORD_CONTEXT_CHARS, max((remaining - len(keyword)) // 2, 0))
        start = max(match.start() - before, last_snippet_end)
        end = min(match.end() + KEYWORD_CONTEXT_CHARS, start + remaining, len(content))
        last_snippet_end = end
        remaining -= end - start
        snippets.append(f"--- 位置 {start}–{end} ---\n{content[start:end]}")

    header = (
        f"[{stored_result.result_id} · {stored_result.tool_name[:80]} 原始结果 · 共 {len(content)} 字 · "
        f"关键词“{keyword}”命中 {total_matches} 处]"
    )
    if not snippets:
        return f"{header}\n（原文中没有这个关键词（从 offset={start_offset} 起），可以换一个词，或不填 keyword 按 offset 顺序读取。）"
    footer = (
        f"\n（片段共享 {max_chars} 字预算；用 offset={last_snippet_end} 继续查找，或不填 keyword 分段读取。）"
        if last_snippet_end < len(content)
        else "\n（已到原文末尾。）"
    )
    return f"{header}\n" + "\n".join(snippets) + footer


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """执行 read_tool_result。"""

    session_id = tool_ctx.runtime.session_id
    if not isinstance(session_id, str) or not session_id:
        return tool_ctx.build_failure_result(invocation.tool_name, "读取原文需要真实聊天 session_id。")
    if any(value and value != session_id for value in (invocation.session_id, context.session_id if context else "")):
        return tool_ctx.build_failure_result(invocation.tool_name, "读取原文的调用与执行上下文 session_id 不一致。")
    result_id = invocation.arguments.get("result_id")
    if not isinstance(result_id, str) or len(result_id) > 64 or not re.fullmatch(r"r[0-9a-f]+", result_id):
        return tool_ctx.build_failure_result(
            invocation.tool_name, "read_tool_result 需要提供工具返回的 result_id（如 r3）。"
        )

    keyword = invocation.arguments.get("keyword") or ""
    if not isinstance(keyword, str) or len(keyword) > READ_MAX_KEYWORD_CHARS:
        return tool_ctx.build_failure_result(
            invocation.tool_name, f"keyword 必须是最多 {READ_MAX_KEYWORD_CHARS} 字的字符串。"
        )
    keyword = keyword.strip()
    stored_result = tool_result_store.get(session_id, result_id)
    if stored_result is None:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"没有找到编号为 {result_id} 的原始结果：它可能已经过期。需要这部分内容时请重新调用原来的工具。",
        )

    offset = _coerce_int(invocation.arguments.get("offset"), 0)
    max_chars = min(
        max(_coerce_int(invocation.arguments.get("max_chars"), READ_DEFAULT_MAX_CHARS), 1), READ_MAX_CHARS_LIMIT
    )
    content = (
        _read_by_keyword(stored_result, keyword, offset, max_chars)
        if keyword
        else _read_by_offset(stored_result, offset, max_chars)
    )
    return tool_ctx.build_success_result(invocation.tool_name, content)
