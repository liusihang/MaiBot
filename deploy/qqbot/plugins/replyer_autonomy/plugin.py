"""Prepare single-Agent requests and supply a trusted dynamic clock."""

from copy import deepcopy
from datetime import datetime
import re
from zoneinfo import ZoneInfo
from collections import deque
from maibot_sdk import Field, HookHandler, MaiBotPlugin, PluginConfigBase


def configure_tools(definitions):
    result = deepcopy(definitions)
    for definition in result:
        function = definition.get("function", definition)
        if function.get("name") not in ("reply", "reply_options"):
            continue
        schema = function.get("parameters", {})
        properties = schema.get("properties", {})
        for name in ("expression_intent", "reply_style"):
            properties.pop(name, None)
        if "required" in schema:
            schema["required"] = [name for name in schema["required"] if name in properties]
        schema["additionalProperties"] = False
        if "reply_reference" in properties:
            properties["reply_reference"]["description"] = (
                "交接回复对象、话题、指代关系、必要事实或工具查询结果。"
                "仅提供背景资料；语气、立场、切入角度、措辞与篇幅由 Replyer 根据人格和现场自主决定。"
            )
    return result


_CLOCK_RE = re.compile(r"\n*^<runtime_clock>\n.*?^</runtime_clock>[ \t]*(?:\n|$)", re.DOTALL | re.MULTILINE)
_TAIL_TIME_RE = re.compile(r"时间：\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\Z")


def inject_runtime_clock(items, now=None):
    """Update trusted system time; retain historical message dates and tool protocol items."""
    result = deepcopy(items)
    system = next((item for item in result if item.get("item_type") == "SystemMessageItem"), None)
    if system is None:
        return result
    part = next((part for part in system.get("parts", []) if part.get("type") == "text"), None)
    if part is None:
        return result
    previous_text = part.get("text", "")
    already_injected = bool(_CLOCK_RE.search(previous_text))
    current = now if now is not None else datetime.now(ZoneInfo("Asia/Shanghai"))
    if current.tzinfo is None:
        raise ValueError("Runtime clock must be timezone-aware")
    current = current.astimezone(ZoneInfo("Asia/Shanghai"))
    weekday = "一二三四五六日"[current.weekday()]
    block = (
        "<runtime_clock>\n"
        f"当前时间：{current:%Y-%m-%d %H:%M:%S}，星期{weekday}。\n"
        "时区：Asia/Shanghai（北京时间，UTC+08:00）。\n"
        "今天、昨天、明天和最近等相对时间以此为准；历史消息和资料按其各自日期理解。\n"
        "</runtime_clock>"
    )
    part["text"] = _CLOCK_RE.sub("", previous_text).rstrip() + "\n\n" + block
    # The core appends one current-time UserMessageItem after history. Remove only
    # that last exact time item on first injection; older day-boundary items remain.
    if not already_injected:
        for index in range(len(result) - 1, -1, -1):
            item = result[index]
            parts = item.get("parts", [])
            if (
                item.get("item_type") == "UserMessageItem"
                and len(parts) == 1
                and parts[0].get("type") == "text"
                and _TAIL_TIME_RE.fullmatch(parts[0].get("text", ""))
            ):
                result.pop(index)
                break
    return result


class RecentExpressions:
    """Previous three selection rounds per chat; empty rounds also age the window."""

    def __init__(self):
        self.rounds = {}

    @staticmethod
    def keys(candidate):
        return {("id", candidate["id"]), ("style", "".join(candidate["style"].split()).casefold())}

    def filter(self, session_id, candidates):
        rounds = self.rounds.setdefault(session_id, deque(maxlen=3))
        excluded = set().union(*rounds) if rounds else set()
        rounds.append(set())
        return [candidate for candidate in candidates if not self.keys(candidate) & excluded]

    def remember(self, session_id, selected):
        rounds = self.rounds.setdefault(session_id, deque(maxlen=3))
        if not rounds:
            rounds.append(set())
        rounds[-1] = set().union(*(self.keys(item) for item in selected)) if selected else set()


class PluginSettings(PluginConfigBase):
    config_version: str = Field(default="1.0.0", description="配置版本")
    enabled: bool = Field(default=True, description="单 Agent 工具约定与动态北京时间")


class Config(PluginConfigBase):
    plugin: PluginSettings = Field(default_factory=PluginSettings)


class ReplyerAutonomyPlugin(MaiBotPlugin):
    config_model = Config

    async def on_load(self):
        self.recent_expressions = RecentExpressions()

    async def on_unload(self):
        pass

    async def on_config_update(self, scope, config_data, version):
        pass

    @HookHandler("expression.select.before_select", description="避免重复最近三轮选过的表达模板")
    async def avoid_recent_expressions(self, **kwargs):
        candidates = self.recent_expressions.filter(kwargs["session_id"], kwargs["candidates"])
        return {"action": "continue", "modified_kwargs": {"candidates": candidates}}

    @HookHandler("expression.select.after_selection", description="记录本聊天本轮选中的表达模板")
    async def remember_selected_expressions(self, **kwargs):
        self.recent_expressions.remember(kwargs["session_id"], kwargs.get("selected_expressions") or [])
        return {"action": "continue"}

    @HookHandler("maisaka.planner.before_request", description="为单 Agent 请求注入动态北京时间")
    async def configure_reply_handoff(self, **kwargs):
        return {
            "action": "continue",
            "modified_kwargs": {
                **kwargs,
                "tool_definitions": configure_tools(kwargs["tool_definitions"]),
                "items": inject_runtime_clock(kwargs["items"]),
            },
        }


def create_plugin() -> ReplyerAutonomyPlugin:
    return ReplyerAutonomyPlugin()
