"""固定消息防抖：安静/活跃开批，同人/换人续发，图片与表情包特殊处理。"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional
import asyncio

DebounceMode = Literal["fast", "natural", "steady"]


@dataclass(frozen=True)
class WaitTimes:
    quiet: float
    active: float
    same_author: float
    different_author: float


WAIT_PROFILES: Dict[DebounceMode, WaitTimes] = {
    "fast": WaitTimes(1.0, 3.0, 1.0, 0.5),
    "natural": WaitTimes(2.0, 5.0, 2.0, 0.5),
    "steady": WaitTimes(4.0, 7.0, 4.0, 2.0),
}

QUIET_THRESHOLD = 30.0
BARE_IMAGE_MIN_WAIT = 4.0
BATCH_LIMIT = 8.0
MENTION_LIMIT = 1.0


class MessageDebouncer:
    """一个运行时一个实例；只管理防抖，不启动后台任务或调用外部接口。"""

    def __init__(self) -> None:
        self._first: Optional[float] = None
        self._last_arrival: Optional[float] = None
        self._last_author: Optional[str] = None
        self._deadline: Optional[float] = None
        self._urgent: Optional[float] = None
        self.changed = asyncio.Event()

    @property
    def deadline(self) -> Optional[float]:
        """当前批次的单调时钟截止时间。"""
        return self._deadline

    def observe(
        self,
        *,
        author: str,
        is_bare_image: bool,
        is_pure_emoji: bool,
        at_bot: bool,
        received_at: float,
        wall_time: float,
        now: float,
        mode: DebounceMode = "natural",
    ) -> str:
        """更新截止时间并返回本次规则说明；预处理耗时也占用 8 秒/1 秒上限。"""
        waits = WAIT_PROFILES[mode]
        arrival = now - max(0.0, wall_time - received_at)
        previous_arrival = self._last_arrival
        previous_author = self._last_author
        if self._last_arrival is not None:
            arrival = max(arrival, self._last_arrival)
        self._last_arrival = arrival
        self._last_author = author
        # Cooldown or an earlier task can leave a long-expired batch unconsumed.
        # A fresh textual burst after silence gets its own bounded continuation
        # window; retain the old messages in the caller's inbox. Existing urgent
        # mentions and emoji-only arrivals keep their earlier admission time.
        reopened = (
            self._deadline is not None
            and arrival >= self._deadline
            and previous_arrival is not None
            and arrival - previous_arrival > QUIET_THRESHOLD
            and self._urgent is None
            and not is_pure_emoji
        )
        if reopened:
            self._first = self._deadline = None
        new_batch = self._first is None
        if new_batch:
            self._first = arrival
        if at_bot:
            urgent = now - max(0.0, wall_time - received_at) + MENTION_LIMIT
            self._urgent = min(self._urgent, urgent) if self._urgent is not None else urgent
        # 同一段活跃连发仍服从原批次上限；长时间安静后的新话头按新窗口收齐。
        # 原批次消息仍在上下文缓存里，消费时一同接入。
        if self._deadline is None or self._deadline > now:
            reasons = []
            limit = self._first + BATCH_LIMIT
            if self._urgent is not None:
                limit = min(limit, self._urgent)
            if new_batch:
                active = previous_arrival is not None and arrival - previous_arrival <= QUIET_THRESHOLD
                wait = waits.active if active else waits.quiet
                reasons.append(f"{'活跃' if active else '安静'}开批 {wait:g} 秒")
                if reopened:
                    wait = max(wait, waits.active)
                    reasons.append(f"到期批次收到新话头，收齐补充 {wait:g} 秒")
                if is_bare_image:
                    reasons.append("裸图开批至少 4 秒")
                    wait = max(wait, BARE_IMAGE_MIN_WAIT)
            else:
                wait = waits.same_author if author == previous_author else waits.different_author
                reasons.append(f"{'同人续发' if author == previous_author else '换人'} {wait:g} 秒")
            candidate = arrival + wait
            # 纯表情包仍可把截止时间提前，但不能在已有批次上继续续长。
            if is_pure_emoji and not new_batch:
                reasons.append("纯表情包不延长等待")
                candidate = min(candidate, self._deadline)
            if candidate > self._first + BATCH_LIMIT:
                reasons.append("受整批 8 秒上限限制")
            if self._urgent is not None and candidate > self._urgent:
                reasons.append("受@或提及 1 秒上限限制")
            self._deadline = min(candidate, limit)
            reason = "；".join(reasons)
        else:
            reason = "批次已到期，不再延长等待"
        self.changed.set()
        return reason

    def consume(self) -> None:
        """本批消息已收走；只保留上一条的时间和发言人，用于下一批判断。"""
        self._first = self._deadline = self._urgent = None
        self.changed.set()
