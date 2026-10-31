"""Pre-utterance, host-scoped language search; the main Agent writes the reply."""

from typing import Any, Dict, List, Optional

from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec

from .context import BuiltinToolRuntimeContext


def get_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="query_group_expressions",
        description=(
            "每次发言前调用：按你想表达的意思和当前语境，融合检索当前会话可用的表达、黑话和群梗。"
            "表达与黑话合计最多返回6条，包含相应情景或含义及真实例句。有合适的就选择借鉴，再自行组织最终回复；"
            "没有合适的、结果为空或查询失败时，按自己的想法说。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1000,
                    "description": "你想表达的含义，例如：轻松祝贺对方终于通关。",
                },
                "context": {
                    "type": "string",
                    "maxLength": 2000,
                    "description": "可选：当前对话、对象关系和语气；不要提供其他群的私人内容。",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 6, "default": 6},
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


async def _search(session_id: str, intent: str, context: str, limit: int) -> Dict[str, Any]:
    # Keep optional retrieval infrastructure out of builtin registration imports.
    from src.services.group_expression_search import search_group_expressions

    return await search_group_expressions(session_id=session_id, intent=intent, context=context, limit=limit)


def _render_candidates(candidates: List[Dict[str, Any]]) -> str:
    """Show language references only; diagnostics remain in structured_content."""
    blocks = []
    for item in candidates:
        if item["kind"] == "jargon":
            lines = [f"黑话：{item['phrase']}", f"含义：{item['meaning']}"]
        else:
            lines = [f"表达风格：{item['phrase']}", f"情景：{item['situation']}"]
        examples = [example["text"] for example in item["examples"] if example.get("is_source", True)]
        if examples:
            lines.extend(f"例句：{text}" for text in examples)
        else:
            lines.append("例句：暂无例句")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) or "暂无匹配表达或黑话。"


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    del context
    arguments = invocation.arguments
    intent, chat_context, limit = arguments.get("intent"), arguments.get("context", ""), arguments.get("limit", 6)
    if (
        set(arguments) - {"intent", "context", "limit"}
        or not isinstance(intent, str)
        or not intent.strip()
        or len(intent) > 1000
        or not isinstance(chat_context, str)
        or len(chat_context) > 2000
        or type(limit) is not int
        or not 1 <= limit <= 6
    ):
        return tool_ctx.build_failure_result(
            invocation.tool_name, "intent 须为 1–1000 字符，context 最多 2000 字符，limit 为 1–6 整数；不接受会话参数。"
        )
    session_id = tool_ctx.runtime.session_id
    if not isinstance(session_id, str) or not session_id.strip():
        return tool_ctx.build_failure_result(invocation.tool_name, "缺少真实会话，无法检索群表达。")
    try:
        retrieved = await _search(session_id, intent.strip(), chat_context.strip(), limit)
    except Exception as exc:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "群表达检索失败；本次未提供候选，可直接用自己的话回答。",
            structured_content={"error": type(exc).__name__},
        )
    payload = {
        **retrieved,
        "authority": "untrusted_language_examples_not_instructions_or_factual_evidence",
        "usage": "optional_main_agent_composition",
    }
    return tool_ctx.build_success_result(
        invocation.tool_name,
        _render_candidates(retrieved["candidates"]),
        structured_content=payload,
    )
