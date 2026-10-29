"""Shared business accounting for actual Pi provider receipts."""

from datetime import datetime
from typing import Any, Dict

import asyncio

from src.common.logger import get_logger
from src.config.model_configs import ModelInfo
from src.llm_models.model_client.base_client import UsageRecord

logger = get_logger("pi_accounting")


async def record_pi_usage(
    payload: Dict[str, Any],
    config: Dict[str, Any],
    started: datetime,
    elapsed: float,
    session_id: str,
    *,
    task_name: str,
    request_type: str,
    succeeded: bool = True,
) -> None:
    route = payload.get("route")
    if not isinstance(route, dict):
        return
    model_data = next((item for item in config["models"] if item["name"] == route["modelName"]), None)
    if model_data is None:
        raise ValueError("Pi returned an unknown configured model identity")
    from src.llm_models.utils import llm_usage_recorder
    from src.services.model_task_usage_service import record_model_task_usage

    model = ModelInfo.model_validate(model_data)
    try:
        await asyncio.to_thread(
            record_model_task_usage,
            task_name,
            model.name,
            route["modelIdentifier"],
            route["apiProvider"],
            payload.get("durationMs", elapsed * 1000) / 1000,
            succeeded,
            started,
        )
        usage = payload.get("usage")
        if isinstance(usage, dict):
            record = UsageRecord(
                model_name=model.name,
                provider_name=route["apiProvider"],
                prompt_tokens=(
                    int(usage.get("input", usage.get("prompt_tokens", usage.get("promptTokenCount", 0))))
                    + int(usage.get("cacheRead", 0))
                    + int(usage.get("cacheWrite", 0))
                ),
                completion_tokens=int(usage.get("output", usage.get("completion_tokens", 0))),
                total_tokens=int(usage.get("totalTokens", usage.get("total_tokens", usage.get("totalTokenCount", 0)))),
                prompt_cache_hit_tokens=int(usage.get("cacheRead", 0)),
                prompt_cache_miss_tokens=int(usage.get("input", 0)) + int(usage.get("cacheWrite", 0)),
                prompt_cache_reported=bool(
                    usage.get("cacheReported", usage.get("cacheRead") or usage.get("cacheWrite"))
                ),
            )
            await asyncio.to_thread(
                llm_usage_recorder.record_usage_to_database,
                model_info=model,
                model_usage=record,
                request_started_at=started,
                user_id="system",
                request_type=request_type,
                task_name=task_name,
                session_id=session_id,
                time_cost=elapsed,
            )
    except Exception:
        # A business statistics failure cannot retry an already completed model.
        logger.exception("记录 Pi 模型调用统计失败")
