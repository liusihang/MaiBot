"""Opt-in daily topic board settings; no independent model transport."""

from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.config.config_base import ConfigBase, Field


class DailyTopicsConfig(ConfigBase):
    __ui_label__ = "最近话题"
    __ui_order__ = 25

    enabled: bool = Field(default=False)
    """启用每群夜间最近话题；不替代实时聊天历史。首次从前一自然日开始采集。"""
    timezone: str = Field(default="Asia/Shanghai")
    """自然日与发布时刻的 IANA 时区；不改变数据库原始消息的本地时间语义。"""
    publish_hour: int = Field(default=3, ge=0, le=23)
    """夜间发布时间的小时；默认北京时间 03:00。"""
    publish_minute: int = Field(default=0, ge=0, le=59)
    """发布时间的分钟；处理此前已经结束的自然日。"""
    model_task: str = Field(default="utils", min_length=1)
    """现有 model_task_config 任务名称；所有摘要调用仍经官方 Pi harness。"""
    max_topics_chars: int = Field(default=2400, ge=500, le=12000)
    """整块话题 JSON 的字符上限（含来源）；超限拒绝，不静默截断。"""
    max_batch_chars: int = Field(default=18000, ge=1000, le=100000)
    """每段原始文本字符预算；长消息按偏移分段而不丢尾，不等于 token 容量证明。"""
    max_batches_per_pass: int = Field(default=8, ge=1, le=100)
    """每群每次调度的模型段数上限；未完成的日任务持久化后下次继续，不发布半成品。"""
    poll_seconds: int = Field(default=60, ge=10, le=3600)
    """后台检查间隔；仅到期日任务进行模型调用，不占用群聊主线程。"""

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("Unknown daily topic timezone") from exc
